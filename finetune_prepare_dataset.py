"""
微调数据集准备脚本
专门为临界区域数据增强设计
"""

import json
import random
import logging
from pathlib import Path
from typing import List, Dict, Optional
import tiktoken

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """计算文本的token数"""
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))
    except:
        # 如果失败，使用简单估算
        return len(text) // 4


def create_prompt(context: str, question: str) -> str:
    """创建prompt"""
    return f"""请阅读以下文本并回答问题。

文本：
{context}

问题：{question}

答案："""


def augment_by_paragraph_shuffle(
    sample: Dict,
    max_context_length: int,
    target_ratio_min: float,
    target_ratio_max: float
) -> List[Dict]:
    """
    通过段落重排增强数据
    
    Args:
        sample: 原始样本
        max_context_length: 最大上下文长度
        target_ratio_min: 目标比率最小值
        target_ratio_max: 目标比率最大值
        
    Returns:
        增强后的样本列表
    """
    augmented = []
    context = sample["context"]
    
    # 按段落分割
    paragraphs = [p.strip() for p in context.split("\n\n") if p.strip()]
    
    if len(paragraphs) < 2:
        return []
    
    # 尝试不同的段落排列组合
    for _ in range(3):  # 生成3个变体
        shuffled_paragraphs = paragraphs.copy()
        random.shuffle(shuffled_paragraphs)
        new_context = "\n\n".join(shuffled_paragraphs)
        
        # 检查是否在目标比率范围内
        prompt = create_prompt(new_context, sample["question"])
        token_count = count_tokens(prompt)
        ratio = token_count / max_context_length
        
        if target_ratio_min <= ratio <= target_ratio_max:
            augmented.append({
                "context": new_context,
                "question": sample["question"],
                "answers": sample["answers"],
                "id": f"{sample.get('id', 'unknown')}_shuffle_{len(augmented)}"
            })
    
    return augmented


def augment_by_context_slicing(
    sample: Dict,
    max_context_length: int,
    target_ratio_min: float,
    target_ratio_max: float
) -> List[Dict]:
    """
    通过上下文切片增强数据（从长文本中截取不同部分）
    
    Args:
        sample: 原始样本
        max_context_length: 最大上下文长度
        target_ratio_min: 目标比率最小值
        target_ratio_max: 目标比率最大值
        
    Returns:
        增强后的样本列表
    """
    augmented = []
    context = sample["context"]
    
    # 如果文本太短，无法切片
    prompt_base = create_prompt("", sample["question"])
    base_tokens = count_tokens(prompt_base)
    available_tokens = int(max_context_length * target_ratio_max) - base_tokens
    
    if len(context) < available_tokens * 4:  # 简单估算
        return []
    
    # 尝试不同的切片位置
    context_tokens = count_tokens(context)
    if context_tokens > available_tokens:
        # 从不同位置开始切片
        for start_ratio in [0.0, 0.2, 0.4, 0.6]:
            start_pos = int(len(context) * start_ratio)
            sliced_context = context[start_pos:start_pos + available_tokens * 4]
            
            prompt = create_prompt(sliced_context, sample["question"])
            token_count = count_tokens(prompt)
            ratio = token_count / max_context_length
            
            if target_ratio_min <= ratio <= target_ratio_max:
                augmented.append({
                    "context": sliced_context,
                    "question": sample["question"],
                    "answers": sample["answers"],
                    "id": f"{sample.get('id', 'unknown')}_slice_{len(augmented)}"
                })
    
    return augmented


def augment_by_question_variation(
    sample: Dict,
    max_context_length: int,
    target_ratio_min: float,
    target_ratio_max: float
) -> List[Dict]:
    """
    通过问题变体增强数据（保持上下文不变）
    
    Args:
        sample: 原始样本
        max_context_length: 最大上下文长度
        target_ratio_min: 目标比率最小值
        target_ratio_max: 目标比率最大值
        
    Returns:
        增强后的样本列表
    """
    # 检查原始样本是否已经在目标范围内
    prompt = create_prompt(sample["context"], sample["question"])
    token_count = count_tokens(prompt)
    ratio = token_count / max_context_length
    
    if target_ratio_min <= ratio <= target_ratio_max:
        # 如果已经在目标范围内，可以生成问题变体
        # 这里简化处理，直接返回原样本（实际可以生成问题变体）
        return [sample.copy()]
    
    return []


def prepare_critical_region_dataset(
    input_data: List[Dict],
    max_context_length: int,
    cliff_start: float = 0.40,
    cliff_end: float = 0.50,
    augmentation_ratio: float = 2.0,
    random_seed: int = 42
) -> List[Dict]:
    """
    准备临界区域数据集
    
    Args:
        input_data: 输入数据
        max_context_length: 最大上下文长度
        cliff_start: 临界区域起始比率
        cliff_end: 临界区域结束比率
        augmentation_ratio: 增强倍数
        random_seed: 随机种子
        
    Returns:
        增强后的数据集
    """
    random.seed(random_seed)
    
    logger.info("="*60)
    logger.info("准备临界区域数据集")
    logger.info("="*60)
    logger.info(f"输入数据量: {len(input_data)}")
    logger.info(f"临界区域: {cliff_start*100:.0f}%-{cliff_end*100:.0f}%")
    logger.info(f"增强倍数: {augmentation_ratio}x")
    
    # 1. 分析数据分布
    logger.info("\n分析数据分布...")
    critical_samples = []
    other_samples = []
    
    for item in input_data:
        prompt = create_prompt(item["context"], item["question"])
        token_count = count_tokens(prompt)
        ratio = token_count / max_context_length
        
        if cliff_start <= ratio <= cliff_end:
            critical_samples.append(item)
        else:
            other_samples.append(item)
    
    logger.info(f"临界区域样本: {len(critical_samples)}")
    logger.info(f"其他区域样本: {len(other_samples)}")
    
    # 2. 计算目标数量
    target_critical_count = int(len(critical_samples) * augmentation_ratio)
    augment_count = max(0, target_critical_count - len(critical_samples))
    
    logger.info(f"\n目标临界区域样本数: {target_critical_count}")
    logger.info(f"需要增强: {augment_count} 个样本")
    
    # 3. 数据增强
    augmented_samples = []
    
    if augment_count > 0:
        logger.info("\n开始数据增强...")
        
        # 策略1: 从长文本中截取到临界区域
        long_samples = [s for s in other_samples 
                       if count_tokens(create_prompt(s["context"], s["question"])) / max_context_length > cliff_end]
        
        logger.info(f"找到 {len(long_samples)} 个长文本样本可用于切片")
        
        for sample in long_samples[:augment_count // 2]:
            augmented = augment_by_context_slicing(
                sample,
                max_context_length,
                cliff_start,
                cliff_end
            )
            augmented_samples.extend(augmented)
            if len(augmented_samples) >= augment_count:
                break
        
        # 策略2: 段落重排
        remaining = augment_count - len(augmented_samples)
        if remaining > 0:
            for sample in critical_samples[:remaining]:
                augmented = augment_by_paragraph_shuffle(
                    sample,
                    max_context_length,
                    cliff_start,
                    cliff_end
                )
                augmented_samples.extend(augmented)
                if len(augmented_samples) >= augment_count:
                    break
        
        # 策略3: 如果还不够，复制现有临界区域样本
        remaining = augment_count - len(augmented_samples)
        if remaining > 0:
            logger.info(f"还需要 {remaining} 个样本，使用复制策略")
            for _ in range(remaining):
                sample = random.choice(critical_samples)
                new_sample = sample.copy()
                new_sample["id"] = f"{sample.get('id', 'unknown')}_copy_{len(augmented_samples)}"
                augmented_samples.append(new_sample)
    
    # 4. 合并数据
    final_data = other_samples + critical_samples + augmented_samples
    
    logger.info("\n" + "="*60)
    logger.info("数据集准备完成")
    logger.info("="*60)
    logger.info(f"最终数据量: {len(final_data)}")
    logger.info(f"  - 其他区域: {len(other_samples)}")
    logger.info(f"  - 原始临界区域: {len(critical_samples)}")
    logger.info(f"  - 增强临界区域: {len(augmented_samples)}")
    
    # 5. 验证增强效果
    logger.info("\n验证增强效果...")
    final_critical = sum(1 for item in final_data 
                        if cliff_start <= count_tokens(create_prompt(item["context"], item["question"])) / max_context_length <= cliff_end)
    logger.info(f"最终临界区域样本数: {final_critical}")
    logger.info(f"增强倍数: {final_critical / len(critical_samples):.2f}x")
    
    return final_data


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="准备微调数据集")
    parser.add_argument("--input-dir", type=str, default="data",
                       help="输入数据目录")
    parser.add_argument("--output-file", type=str, default="data_finetune/finetune_dataset.json",
                       help="输出文件路径")
    parser.add_argument("--max-context-length", type=int, default=131072,
                       help="最大上下文长度")
    parser.add_argument("--cliff-start", type=float, default=0.40,
                       help="临界区域起始比率")
    parser.add_argument("--cliff-end", type=float, default=0.50,
                       help="临界区域结束比率")
    parser.add_argument("--augmentation-ratio", type=float, default=2.0,
                       help="增强倍数")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    
    args = parser.parse_args()
    
    # 加载数据
    logger.info("加载原始数据...")
    input_data = []
    
    data_dir = Path(args.input_dir)
    
    # 加载SQuAD
    squad_path = data_dir / "squad.json"
    if squad_path.exists():
        with open(squad_path, 'r', encoding='utf-8') as f:
            squad_data = json.load(f)
            input_data.extend(squad_data)
            logger.info(f"加载SQuAD: {len(squad_data)} 条")
    
    # 加载NarrativeQA
    narrativeqa_path = data_dir / "narrativeqa.json"
    if narrativeqa_path.exists():
        with open(narrativeqa_path, 'r', encoding='utf-8') as f:
            narrativeqa_data = json.load(f)
            input_data.extend(narrativeqa_data)
            logger.info(f"加载NarrativeQA: {len(narrativeqa_data)} 条")
    
    if not input_data:
        logger.error("未找到数据文件！请先运行 prepare_datasets.py")
        return
    
    # 准备数据集
    final_data = prepare_critical_region_dataset(
        input_data,
        args.max_context_length,
        args.cliff_start,
        args.cliff_end,
        args.augmentation_ratio,
        args.seed
    )
    
    # 保存
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n数据集已保存到: {output_path}")


if __name__ == "__main__":
    main()

