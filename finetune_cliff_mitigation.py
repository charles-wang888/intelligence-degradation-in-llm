"""
断崖式降智缓解微调脚本
实现临界区域加权损失和RoPE参数微调
"""

import os
import json
import logging
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import tiktoken
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class FineTuneConfig:
    """微调配置"""
    # 模型配置
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    max_context_length: int = 131072  # 128K
    
    # 临界区域配置（基于论文发现）
    cliff_region_start: float = 0.40  # 40%
    cliff_region_end: float = 0.50    # 50%
    critical_weight: float = 3.0       # 临界区域权重倍数
    
    # 数据配置
    data_dir: str = "data"
    output_dir: str = "finetune_output"
    train_split: float = 0.9
    
    # 训练配置
    batch_size: int = 1  # 长上下文需要小batch
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 100
    max_steps: Optional[int] = None
    
    # RoPE微调配置
    enable_rope_tuning: bool = True
    rope_base: Optional[float] = None  # None表示使用默认值
    
    # 数据增强配置
    enable_data_augmentation: bool = True
    augmentation_ratio: float = 2.0  # 临界区域数据增强倍数
    
    # 随机种子
    seed: int = 42


class CriticalRegionDataset(Dataset):
    """临界区域数据集，支持加权采样"""
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_length: int,
        cliff_start: float,
        cliff_end: float,
        critical_weight: float,
        use_fast_token_count: bool = True
    ):
        """
        初始化数据集
        
        Args:
            data: 数据列表，每个元素包含 context, question, answers
            tokenizer: tokenizer
            max_length: 最大长度
            cliff_start: 临界区域起始比率
            cliff_end: 临界区域结束比率
            critical_weight: 临界区域权重
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cliff_start = cliff_start
        self.cliff_end = cliff_end
        self.critical_weight = critical_weight
        
        # 计算每个样本的上下文长度比率和权重
        self.sample_weights = []
        self.context_ratios = []
        self.valid_indices = []  # 记录有效样本的索引
        
        # 预留一些token给特殊token（如eos_token等）
        reserved_tokens = 10
        effective_max_length = max_length - reserved_tokens
        
        logger.info(f"过滤超长样本（最大长度: {max_length}, 有效长度: {effective_max_length}）...")
        logger.info(f"处理 {len(data)} 个样本，这可能需要一些时间...")
        
        # 初始化快速token计数器（使用tiktoken）
        fast_tokenizer = None
        if use_fast_token_count:
            try:
                # 尝试使用tiktoken进行快速计数
                # 对于Qwen模型，通常使用cl100k_base编码
                fast_tokenizer = tiktoken.get_encoding("cl100k_base")
                logger.info("使用 tiktoken 进行快速token计数")
            except Exception as e:
                logger.warning(f"无法加载tiktoken: {e}，将使用tokenizer进行计数（较慢）")
                fast_tokenizer = None
        
        # 使用tqdm显示进度
        try:
            from tqdm import tqdm
            iterator = tqdm(enumerate(data), total=len(data), desc="处理样本")
        except ImportError:
            logger.warning("tqdm未安装，无法显示进度条。建议安装: pip install tqdm")
            iterator = enumerate(data)
        
        last_log_time = 0
        import time
        start_time = time.time()
        
        for idx, item in iterator:
            # 每10秒输出一次进度（即使tqdm也在更新）
            current_time = time.time()
            if current_time - last_log_time > 10:
                elapsed = current_time - start_time
                rate = (idx + 1) / elapsed if elapsed > 0 else 0
                remaining = (len(data) - idx - 1) / rate if rate > 0 else 0
                logger.info(f"进度: {idx + 1}/{len(data)} ({idx*100//len(data)}%), "
                          f"已用时: {elapsed:.1f}秒, 预计剩余: {remaining:.1f}秒, "
                          f"有效样本: {len(self.valid_indices)}")
                last_log_time = current_time
            
            # 构造prompt
            prompt = self._create_prompt(item)
            
            # 快速预检查：如果文本明显超长，直接跳过（避免tokenize）
            # 简单估算：1 token ≈ 3-4字符（保守估计）
            estimated_tokens = len(prompt) // 3
            if estimated_tokens > effective_max_length * 1.2:  # 1.2倍缓冲
                continue
            
            # 计算token数（优先使用tiktoken快速计数）
            try:
                if fast_tokenizer is not None:
                    # 使用tiktoken快速计数
                    token_count = len(fast_tokenizer.encode(prompt))
                else:
                    # 回退到tokenizer（较慢）
                    # 对于明显很长的文本，先截断再tokenize（加快速度）
                    if len(prompt) > effective_max_length * 4:  # 如果文本长度超过4倍，先截断
                        prompt = prompt[:effective_max_length * 4]
                    
                    tokens = tokenizer.encode(
                        prompt, 
                        add_special_tokens=False,
                        max_length=effective_max_length,
                        truncation=True
                    )
                    token_count = len(tokens)
            except Exception as e:
                if idx % 1000 == 0:  # 每1000个样本输出一次警告
                    logger.warning(f"样本 {idx} tokenize失败: {e}，跳过")
                continue
            
            # 如果超过最大长度，跳过该样本
            if token_count > effective_max_length:
                continue
            
            # 计算上下文长度比率
            ratio = token_count / max_length
            self.context_ratios.append(ratio)
            
            # 计算权重：临界区域样本权重更高
            if cliff_start <= ratio <= cliff_end:
                weight = critical_weight
            else:
                weight = 1.0
            self.sample_weights.append(weight)
            self.valid_indices.append(idx)
        
        elapsed_total = time.time() - start_time
        logger.info(f"样本处理完成！有效样本: {len(self.valid_indices)}/{len(data)}, 总用时: {elapsed_total:.1f}秒")
        
        # 验证数据一致性
        if len(self.valid_indices) != len(self.context_ratios) or len(self.valid_indices) != len(self.sample_weights):
            raise ValueError(
                f"数据不一致！valid_indices: {len(self.valid_indices)}, "
                f"context_ratios: {len(self.context_ratios)}, "
                f"sample_weights: {len(self.sample_weights)}"
            )
        
        # 只保留有效样本的数据
        self.data = [self.data[i] for i in self.valid_indices]
        # context_ratios 和 sample_weights 已经是按 valid_indices 顺序添加的，不需要重新索引
        
        # 归一化权重用于采样
        if len(self.sample_weights) > 0:
            self.sample_weights = np.array(self.sample_weights)
            self.sample_weights = self.sample_weights / self.sample_weights.sum()
        else:
            raise ValueError("没有有效的样本！所有样本都超过了最大长度。")
        
        logger.info(f"数据集大小: {len(self.data)} (原始: {len(data)}, 过滤: {len(data) - len(self.data)})")
        logger.info(f"临界区域样本数: {sum(1 for r in self.context_ratios if cliff_start <= r <= cliff_end)}")
        logger.info(f"平均权重: {np.mean(self.sample_weights):.4f}")
        
        # 验证最终数据一致性
        assert len(self.data) == len(self.context_ratios) == len(self.sample_weights), \
            f"最终数据不一致！data: {len(self.data)}, ratios: {len(self.context_ratios)}, weights: {len(self.sample_weights)}"
    
    def _create_prompt(self, item: Dict) -> str:
        """创建prompt"""
        context = item.get("context", "")
        question = item.get("question", "")
        
        prompt = f"""请阅读以下文本并回答问题。

文本：
{context}

问题：{question}

答案："""
        return prompt
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 边界检查
        if idx < 0 or idx >= len(self.data):
            raise IndexError(f"索引 {idx} 超出范围 [0, {len(self.data)})")
        
        item = self.data[idx]
        prompt = self._create_prompt(item)
        
        # Tokenize（不添加特殊token，因为会在collator中添加）
        # 预留一些token给特殊token
        reserved_tokens = 10
        effective_max_length = self.max_length - reserved_tokens
        
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=effective_max_length,  # 使用有效最大长度
            padding=False,  # 不在dataset中padding，在collator中统一处理
            return_tensors=None,  # 返回list而不是tensor
            add_special_tokens=False  # 不在tokenize时添加特殊token
        )
        
        # 再次检查长度（防止意外超长）
        if len(encoding["input_ids"]) > effective_max_length:
            logger.warning(f"样本 {idx} 在tokenize后仍然超长，强制截断")
            encoding["input_ids"] = encoding["input_ids"][:effective_max_length]
            encoding["attention_mask"] = encoding["attention_mask"][:effective_max_length]
        
        # 获取上下文长度比率和权重（确保索引在范围内）
        if idx >= len(self.context_ratios) or idx >= len(self.sample_weights):
            raise IndexError(
                f"索引 {idx} 超出范围！"
                f"data长度: {len(self.data)}, "
                f"ratios长度: {len(self.context_ratios)}, "
                f"weights长度: {len(self.sample_weights)}"
            )
        
        ratio = self.context_ratios[idx]
        weight = self.sample_weights[idx]
        
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "context_ratio": float(ratio),  # 确保是Python float类型
            "weight": float(weight)  # 确保是Python float类型
        }


class WeightedTrainer(Trainer):
    """支持加权损失的Trainer"""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        计算加权损失
        
        Args:
            model: 模型
            inputs: 输入数据，包含 weight 字段
            return_outputs: 是否返回输出
            num_items_in_batch: 批次中的项目数（新版本transformers会传递此参数，忽略即可）
        """
        # 获取权重（需要先保存，因为pop会删除）
        weights = inputs.get("weight", None)
        context_ratios = inputs.get("context_ratio", None)
        
        # 创建新的inputs字典，移除weight和context_ratio（模型不需要这些）
        model_inputs = {k: v for k, v in inputs.items() 
                       if k not in ["weight", "context_ratio"]}
        
        # 标准前向传播
        labels = model_inputs.get("labels")
        if labels is None:
            # 如果没有labels，使用input_ids作为labels（语言模型训练）
            labels = model_inputs["input_ids"].clone()
            model_inputs["labels"] = labels
        
        outputs = model(**model_inputs)
        logits = outputs.logits
        
        # 对于长上下文，需要优化损失计算以避免显存爆炸
        # 使用分块计算，避免一次性转换整个logits为float32
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # 获取序列长度和批次大小
        batch_size, seq_len, vocab_size = shift_logits.shape
        
        # 对于超长序列，分块计算损失
        # 每块处理一定数量的token，避免显存爆炸
        # 对于131K上下文，使用更小的chunk_size（2048）以确保显存足够
        chunk_size = min(2048, seq_len)  # 每次处理最多2048个token（更保守）
        
        if seq_len > chunk_size:
            # 分块计算损失
            logger.debug(f"序列长度 {seq_len} 超过 {chunk_size}，使用分块计算损失")
            per_token_loss_chunks = []
            
            for i in range(0, seq_len, chunk_size):
                end_idx = min(i + chunk_size, seq_len)
                chunk_logits = shift_logits[:, i:end_idx, :]
                chunk_labels = shift_labels[:, i:end_idx]
                
                # 计算这一块的损失
                loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
                chunk_loss = loss_fct(
                    chunk_logits.view(-1, vocab_size).float(),  # 只转换这一块为float32
                    chunk_labels.view(-1)
                )
                per_token_loss_chunks.append(chunk_loss.view(batch_size, end_idx - i))
            
            # 拼接所有块
            per_token_loss = torch.cat(per_token_loss_chunks, dim=1)
        else:
            # 序列不太长，直接计算
            loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
            per_token_loss = loss_fct(
                shift_logits.view(-1, vocab_size).float(),  # 转换为float32计算损失
                shift_labels.view(-1)
            )
            # 重塑为 [batch_size, seq_len]
            per_token_loss = per_token_loss.view(shift_labels.shape)
        
        # 应用attention mask（忽略padding和-100）
        attention_mask = model_inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask[..., 1:].contiguous()
            # 同时忽略label为-100的位置
            valid_mask = (shift_labels != -100) & (attention_mask == 1)
            per_token_loss = per_token_loss * valid_mask.float()
            
            # 计算每个样本的平均损失
            per_sample_loss = per_token_loss.sum(dim=1) / valid_mask.sum(dim=1).float().clamp(min=1.0)
        else:
            # 如果没有attention_mask，只忽略-100
            valid_mask = (shift_labels != -100)
            per_token_loss = per_token_loss * valid_mask.float()
            per_sample_loss = per_token_loss.sum(dim=1) / valid_mask.sum(dim=1).float().clamp(min=1.0)
        
        # 应用权重
        if weights is not None:
            # 确保weights在正确的设备上
            if isinstance(weights, torch.Tensor):
                weights = weights.to(per_sample_loss.device)
            weighted_loss = (per_sample_loss * weights).mean()
        else:
            weighted_loss = per_sample_loss.mean()
        
        return (weighted_loss, outputs) if return_outputs else weighted_loss


def augment_critical_region_data(
    data: List[Dict],
    tokenizer,
    max_context_length: int,
    cliff_start: float,
    cliff_end: float,
    augmentation_ratio: float
) -> List[Dict]:
    """
    增强临界区域的数据
    
    Args:
        data: 原始数据
        tokenizer: tokenizer
        max_context_length: 最大上下文长度
        cliff_start: 临界区域起始比率
        cliff_end: 临界区域结束比率
        augmentation_ratio: 增强倍数
        
    Returns:
        增强后的数据列表
    """
    logger.info("开始临界区域数据增强...")
    
    # 预留一些token给特殊token
    reserved_tokens = 10
    effective_max_length = max_context_length - reserved_tokens
    
    # 初始化快速token计数器（使用tiktoken）
    fast_tokenizer = None
    try:
        # 尝试使用tiktoken进行快速计数
        fast_tokenizer = tiktoken.get_encoding("cl100k_base")
        logger.info("使用 tiktoken 进行快速token计数")
    except Exception as e:
        logger.warning(f"无法加载tiktoken: {e}，将使用tokenizer进行计数（较慢）")
        fast_tokenizer = None
    
    # 找出临界区域的样本
    critical_samples = []
    skipped_count = 0
    
    logger.info(f"分析 {len(data)} 个样本，查找临界区域样本...")
    
    # 使用tqdm显示进度
    try:
        from tqdm import tqdm
        iterator = tqdm(enumerate(data), total=len(data), desc="分析样本")
    except ImportError:
        iterator = enumerate(data)
    
    import time
    start_time = time.time()
    last_log_time = 0
    
    for idx, item in iterator:
        # 每10秒输出一次进度
        current_time = time.time()
        if current_time - last_log_time > 10:
            elapsed = current_time - start_time
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            remaining = (len(data) - idx - 1) / rate if rate > 0 else 0
            logger.info(f"分析进度: {idx + 1}/{len(data)} ({idx*100//len(data)}%), "
                      f"已用时: {elapsed:.1f}秒, 预计剩余: {remaining:.1f}秒, "
                      f"找到临界样本: {len(critical_samples)}")
            last_log_time = current_time
        
        prompt = f"""请阅读以下文本并回答问题。

文本：
{item.get("context", "")}

问题：{item.get("question", "")}

答案："""
        
        # 快速预检查
        estimated_tokens = len(prompt) // 3
        if estimated_tokens > effective_max_length * 1.2:
            skipped_count += 1
            continue
        
        try:
            # 使用tiktoken快速计数或tokenizer计数
            if fast_tokenizer is not None:
                token_count = len(fast_tokenizer.encode(prompt))
            else:
                # 对于明显很长的文本，先截断
                if len(prompt) > effective_max_length * 4:
                    prompt = prompt[:effective_max_length * 4]
                
                tokens = tokenizer.encode(
                    prompt, 
                    add_special_tokens=False,
                    max_length=effective_max_length,
                    truncation=True
                )
                token_count = len(tokens)
            
            # 如果超过有效最大长度，跳过
            if token_count > effective_max_length:
                skipped_count += 1
                continue
            
            ratio = token_count / max_context_length
            
            if cliff_start <= ratio <= cliff_end:
                critical_samples.append(item)
        except Exception as e:
            if idx % 1000 == 0:
                logger.warning(f"处理样本 {idx} 时出错: {e}，跳过")
            skipped_count += 1
            continue
    
    if skipped_count > 0:
        logger.info(f"跳过了 {skipped_count} 个超长样本")
    
    logger.info(f"找到 {len(critical_samples)} 个临界区域样本")
    
    # 计算需要增强的数量
    target_count = int(len(critical_samples) * augmentation_ratio)
    augment_count = target_count - len(critical_samples)
    
    if augment_count <= 0:
        logger.info("临界区域样本已足够，无需增强")
        return data
    
    logger.info(f"需要增强 {augment_count} 个样本")
    
    # 数据增强策略
    augmented_samples = []
    
    for _ in range(augment_count):
        # 随机选择一个临界区域样本
        base_sample = random.choice(critical_samples)
        
        # 策略1: 段落重排（保持长度在临界区域）
        context = base_sample["context"]
        paragraphs = context.split("\n\n")
        
        if len(paragraphs) > 1:
            # 随机重排段落
            random.shuffle(paragraphs)
            new_context = "\n\n".join(paragraphs)
            
            augmented_samples.append({
                "context": new_context,
                "question": base_sample["question"],
                "answers": base_sample["answers"],
                "id": f"{base_sample.get('id', 'unknown')}_aug_{len(augmented_samples)}"
            })
        else:
            # 策略2: 如果无法重排，直接复制（增加该样本的权重）
            augmented_samples.append(base_sample.copy())
    
    # 合并原始数据和增强数据
    augmented_data = data + augmented_samples
    
    logger.info(f"数据增强完成: {len(data)} -> {len(augmented_data)}")
    
    return augmented_data


def load_training_data(config: FineTuneConfig) -> List[Dict]:
    """加载训练数据"""
    logger.info("加载训练数据...")
    
    data_dir = Path(config.data_dir)
    all_data = []
    
    # 优先尝试加载预处理好的数据集
    finetune_dataset_path = data_dir / "finetune_dataset.json"
    if finetune_dataset_path.exists():
        logger.info(f"找到预处理数据集: {finetune_dataset_path}")
        with open(finetune_dataset_path, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
            logger.info(f"加载预处理数据集: {len(all_data)} 条")
            return all_data
    
    # 如果没有预处理数据集，从原始数据加载
    logger.info("未找到预处理数据集，从原始数据加载...")
    
    # 尝试从data_finetune目录加载
    if data_dir.name == "data_finetune":
        # 如果data_finetune目录没有原始数据，尝试从data目录加载
        data_dir_alt = Path("data")
        if (data_dir_alt / "squad.json").exists() or (data_dir_alt / "narrativeqa.json").exists():
            logger.info(f"从 {data_dir_alt} 目录加载原始数据...")
            data_dir = data_dir_alt
    
    # 加载SQuAD数据
    squad_path = data_dir / "squad.json"
    if squad_path.exists():
        with open(squad_path, 'r', encoding='utf-8') as f:
            squad_data = json.load(f)
            all_data.extend(squad_data)
            logger.info(f"加载SQuAD数据: {len(squad_data)} 条")
    
    # 加载NarrativeQA数据
    narrativeqa_path = data_dir / "narrativeqa.json"
    if narrativeqa_path.exists():
        with open(narrativeqa_path, 'r', encoding='utf-8') as f:
            narrativeqa_data = json.load(f)
            all_data.extend(narrativeqa_data)
            logger.info(f"加载NarrativeQA数据: {len(narrativeqa_data)} 条")
    
    if not all_data:
        logger.error(f"未找到数据文件！请检查 {data_dir} 目录")
        raise FileNotFoundError(f"未找到数据文件在 {data_dir}")
    
    logger.info(f"总数据量: {len(all_data)} 条")
    
    # 数据增强
    if config.enable_data_augmentation:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        all_data = augment_critical_region_data(
            all_data,
            tokenizer,
            config.max_context_length,
            config.cliff_region_start,
            config.cliff_region_end,
            config.augmentation_ratio
        )
    
    return all_data


def setup_rope_tuning(model, rope_base: Optional[float] = None):
    """
    设置RoPE参数微调
    
    Args:
        model: 模型
        rope_base: RoPE base频率（None表示使用默认值）
    """
    logger.info("设置RoPE参数微调...")
    
    # 查找RoPE相关的层
    rope_params = []
    
    for name, module in model.named_modules():
        if "rotary" in name.lower() or "rope" in name.lower():
            rope_params.append((name, module))
            logger.info(f"找到RoPE层: {name}")
    
    if not rope_params:
        logger.warning("未找到RoPE层，将尝试查找embedding层")
        # 对于Qwen2.5，RoPE可能在embedding或attention层中
        for name, module in model.named_modules():
            if "embed" in name.lower() and hasattr(module, "inv_freq"):
                rope_params.append((name, module))
                logger.info(f"找到可能的RoPE相关层: {name}")
    
    if rope_params:
        logger.info(f"找到 {len(rope_params)} 个RoPE相关层")
        # 启用这些参数的梯度
        for name, module in rope_params:
            for param in module.parameters():
                param.requires_grad = True
                logger.info(f"启用 {name} 的梯度")
    else:
        logger.warning("未找到RoPE层，RoPE微调可能无法生效")
    
    # 如果指定了rope_base，尝试设置
    if rope_base is not None:
        logger.info(f"设置RoPE base为: {rope_base}")
        # 这里需要根据具体模型架构来设置
        # 对于Qwen2.5，可能需要修改embedding层的inv_freq


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="断崖式降智缓解微调")
    parser.add_argument("--model", type=str, default=r"D:\Development\LLM-已下载的大模型\Models\HuggingFace\Qwen\Qwen2.5-7B",
                       help="模型名称或本地路径（例如：models/qwen2.5-7b-instruct）")
    parser.add_argument("--data-dir", type=str, default="data_finetune",
                       help="数据目录")
    parser.add_argument("--output-dir", type=str, default="finetune_output",
                       help="输出目录")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="批次大小")
    parser.add_argument("--learning-rate", type=float, default=2e-5,
                       help="学习率")
    parser.add_argument("--num-epochs", type=int, default=3,
                       help="训练轮数")
    parser.add_argument("--critical-weight", type=float, default=3.0,
                       help="临界区域权重")
    parser.add_argument("--enable-rope-tuning", action="store_true",
                       help="启用RoPE微调")
    parser.add_argument("--enable-data-aug", action="store_true", default=True,
                       help="启用数据增强")
    parser.add_argument("--augmentation-ratio", type=float, default=2.0,
                       help="数据增强倍数")
    
    args = parser.parse_args()
    
    # 创建配置
    config = FineTuneConfig(
        model_name=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        critical_weight=args.critical_weight,
        enable_rope_tuning=args.enable_rope_tuning,
        enable_data_augmentation=args.enable_data_aug,
        augmentation_ratio=args.augmentation_ratio
    )
    
    # 设置随机种子
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    logger.info("="*60)
    logger.info("断崖式降智缓解微调")
    logger.info("="*60)
    logger.info(f"模型: {config.model_name}")
    logger.info(f"临界区域: {config.cliff_region_start*100:.0f}%-{config.cliff_region_end*100:.0f}%")
    logger.info(f"临界区域权重: {config.critical_weight}x")
    logger.info(f"RoPE微调: {config.enable_rope_tuning}")
    logger.info(f"数据增强: {config.enable_data_augmentation}")
    logger.info("="*60)
    
    # 加载数据
    data = load_training_data(config)
    
    # 加载模型和tokenizer
    logger.info(f"加载模型: {config.model_name}")
    
    # 检查是本地路径还是HuggingFace模型名称
    model_path = Path(config.model_name)
    if model_path.exists() and model_path.is_dir():
        logger.info(f"使用本地模型: {model_path}")
        local_model = True
    else:
        logger.info(f"从HuggingFace下载模型: {config.model_name}")
        local_model = False
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        logger.info("✓ 模型加载成功")
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        if not local_model:
            logger.error("提示：如果网络问题，请先运行以下命令下载模型到本地：")
            logger.error(f"  python finetune_download_model.py --model-name {config.model_name}")
        raise
    
    # 启用gradient checkpointing（如果模型支持）
    # 注意：gradient checkpointing 与 use_cache 不兼容，需要先禁用 use_cache
    if hasattr(model, "config"):
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
            logger.info("已禁用 use_cache（gradient checkpointing 需要）")
    
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("已启用gradient checkpointing以节省内存")
    
    # 设置RoPE微调
    if config.enable_rope_tuning:
        setup_rope_tuning(model, config.rope_base)
    
    # 创建数据集
    logger.info("创建数据集...")
    dataset = CriticalRegionDataset(
        data,
        tokenizer,
        config.max_context_length,
        config.cliff_region_start,
        config.cliff_region_end,
        config.critical_weight
    )
    
    # 划分训练集和验证集
    # 使用自定义的Subset包装类，确保索引映射正确
    train_size = int(len(dataset) * config.train_split)
    
    class IndexedSubset(torch.utils.data.Dataset):
        """自定义Subset，确保索引映射正确"""
        def __init__(self, dataset, indices):
            self.dataset = dataset
            self.indices = list(indices)  # 确保是列表
        
        def __len__(self):
            return len(self.indices)
        
        def __getitem__(self, idx):
            # 使用原始数据集的索引
            if idx >= len(self.indices):
                raise IndexError(f"Subset索引 {idx} 超出范围 [0, {len(self.indices)})")
            original_idx = self.indices[idx]
            result = self.dataset[original_idx]
            # 验证返回的字典包含必要的字段
            if not isinstance(result, dict):
                raise ValueError(f"数据集返回的不是字典: {type(result)}")
            if "weight" not in result or "context_ratio" not in result:
                raise ValueError(
                    f"数据集返回的字典缺少必要字段！"
                    f"原始索引: {original_idx}, Subset索引: {idx}, "
                    f"返回的键: {list(result.keys())}"
                )
            return result
    
    train_dataset = IndexedSubset(dataset, list(range(train_size)))
    val_dataset = IndexedSubset(dataset, list(range(train_size, len(dataset))))
    
    logger.info(f"训练集: {len(train_dataset)} 条")
    logger.info(f"验证集: {len(val_dataset)} 条")
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        logging_steps=10,
        save_steps=500,
        eval_steps=500,
        eval_strategy="steps",  # 使用新的参数名，避免deprecation警告
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=False,
        bf16=True,
        dataloader_pin_memory=False,
        report_to="none",
        # 内存优化选项
        gradient_checkpointing=True,  # 启用gradient checkpointing以节省内存
        dataloader_num_workers=0,  # 减少数据加载器的工作进程数（Windows上可能有问题）
        remove_unused_columns=False,  # 保留所有列（包括weight和context_ratio）
        max_grad_norm=1.0,  # 梯度裁剪
        # 额外的内存优化
        dataloader_drop_last=True,  # 丢弃最后一个不完整的batch
        prediction_loss_only=True,  # 只计算损失，不计算其他指标
        include_inputs_for_metrics=False  # 不包含输入用于指标计算
    )
    
    # 自定义数据整理器，保留weight和context_ratio
    class WeightedDataCollator(DataCollatorForLanguageModeling):
        def __call__(self, features):
            # 提取weight和context_ratio（使用get避免KeyError）
            weights = []
            context_ratios = []
            missing_count = 0
            for i, f in enumerate(features):
                if not isinstance(f, dict):
                    raise ValueError(f"特征 {i} 不是字典类型: {type(f)}")
                
                weight = f.pop("weight", None)
                context_ratio = f.pop("context_ratio", None)
                
                if weight is None or context_ratio is None:
                    missing_count += 1
                    if missing_count <= 3:  # 只记录前3个缺失的样本
                        logger.warning(
                            f"样本 {i} 缺少字段 - weight: {weight is not None}, "
                            f"context_ratio: {context_ratio is not None}, "
                            f"可用键: {list(f.keys())}"
                        )
                
                if weight is not None:
                    weights.append(weight)
                else:
                    # 如果没有weight，使用默认值1.0
                    weights.append(1.0)
                
                if context_ratio is not None:
                    context_ratios.append(context_ratio)
                else:
                    # 如果没有context_ratio，使用默认值0.0
                    context_ratios.append(0.0)
            
            if missing_count > 0:
                logger.warning(f"共有 {missing_count}/{len(features)} 个样本缺少weight或context_ratio字段")
            
            # 调用父类方法处理其他字段
            batch = super().__call__(features)
            
            # 添加weight和context_ratio回batch
            batch["weight"] = torch.tensor(weights, dtype=torch.float32)
            batch["context_ratio"] = torch.tensor(context_ratios, dtype=torch.float32)
            
            return batch
    
    data_collator = WeightedDataCollator(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # 创建Trainer
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=tokenizer  # 使用新的参数名，避免deprecation警告
    )
    
    # 开始训练
    logger.info("开始训练...")
    trainer.train()
    
    # 保存模型
    logger.info(f"保存模型到: {config.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)
    
    logger.info("训练完成！")


if __name__ == "__main__":
    main()

