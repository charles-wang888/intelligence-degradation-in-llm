"""
自然长度分布实验
使用每个样本的自然token长度，不进行任何截断或填充
分析：自然长度比率 vs. 性能指标的关系
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import tiktoken

from config import (
    QWEN_MODELS,
    TASK_TYPES,
    DATASETS,
    EXPERIMENT_CONFIG,
    RESULTS_CONFIG,
    OLLAMA_CONFIG,
    VLLM_CONFIG,
    get_model_max_context
)
from utils import OllamaClient, VLLMClient
from utils.dataset_loader import get_dataset_loader
from utils.evaluator import get_evaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class NaturalLengthExperimentRunner:
    """基于自然长度分布的实验运行器"""

    def __init__(
        self,
        results_dir: str = "results",
        ollama_url: str = OLLAMA_CONFIG["base_url"],
        vllm_url: str = VLLM_CONFIG["base_url"],
        vllm_api_key: Optional[str] = None,
        llm_backend: str = "ollama",
        dataset_name: str = None
    ):
        """
        初始化实验运行器

        Args:
            results_dir: 结果保存目录
            ollama_url: Ollama服务地址
            vllm_url: vLLM服务基地址（OpenAI兼容接口）
            vllm_api_key: vLLM API key，放在Authorization头里
            llm_backend: 服务类型，支持 ollama 或 vllm
            dataset_name: 数据集名称
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.dataset_name = dataset_name

        if dataset_name:
            self.dataset_dir = self.results_dir / dataset_name
            self.dataset_dir.mkdir(exist_ok=True)
            logger.info(f"结果将保存到: {self.dataset_dir}")

            self.logs_dir = Path(RESULTS_CONFIG["logs_dir"]) / dataset_name
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"日志将保存到: {self.logs_dir}")
        else:
            self.dataset_dir = self.results_dir
            self.logs_dir = Path(RESULTS_CONFIG["logs_dir"])
            self.logs_dir.mkdir(exist_ok=True)

        self.llm_backend = (llm_backend or "ollama").lower()
        if self.llm_backend not in {"ollama", "vllm"}:
            logger.warning(f"不支持的后端 {self.llm_backend}，使用 ollama")
            self.llm_backend = "ollama"

        if self.llm_backend == "vllm":
            self.llm_client = VLLMClient(base_url=vllm_url, api_key=vllm_api_key, timeout=EXPERIMENT_CONFIG["timeout"])
            self.base_url = vllm_url
        else:
            self.llm_client = OllamaClient(base_url=ollama_url, timeout=EXPERIMENT_CONFIG["timeout"])
            self.base_url = ollama_url
        logger.info(f"LLM后端: {self.llm_backend}, 地址: {self.base_url}")

        self.results = []

        # 初始化tokenizer（用于计算token数）
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            logger.info("已加载 tiktoken tokenizer")
        except Exception as e:
            logger.warning(f"无法加载 tiktoken: {e}，将使用近似方法计算token数")
            self.tokenizer = None

    def count_tokens(self, text: str) -> int:
        """
        计算文本的token数

        Args:
            text: 输入文本

        Returns:
            token数量
        """
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception as e:
                logger.warning(f"tokenizer计算失败: {e}，使用近似方法")

        # 备用方法：近似估算
        # 英文: ~4 chars per token, 中文: ~1.5 chars per token
        # 使用保守估计：2.5 chars per token
        return len(text) // 2

    def get_model_name(self, model_info: Dict, model_key: str) -> str:
        """
        根据后端返回正确的模型名称。
        """
        if self.llm_backend == "vllm":
            return model_info.get("vllm_name") or model_key
        return model_info.get("ollama_name") or model_key

    def build_prompt(self, task_type: str, context: str, question: str = "") -> str:
        """
        构建任务提示词

        Args:
            task_type: 任务类型
            context: 上下文文本
            question: 问题（某些任务需要）

        Returns:
            完整的提示词
        """
        task_templates = {
            "reading_comprehension": """请阅读以下文本并回答问题。

文本：
{context}

问题：{question}

请直接给出答案，不需要解释。""",

            "summarization": """请阅读以下文本，生成简洁准确的摘要。

文本：
{context}

请总结文本的主要内容，生成一个简洁的摘要。"""
        }

        template = task_templates.get(task_type, task_templates["reading_comprehension"])
        return template.format(context=context, question=question)

    def run_single_sample(self,
                          model_name: str,
                          model_key: str,
                          dataset_name: str,
                          task_type: str,
                          sample: Dict,
                          max_context_tokens: int) -> Dict:
        """
        运行单个样本的实验（使用自然长度）

        Args:
            model_name: 模型名称（按后端要求格式）
            model_key: 模型键名
            dataset_name: 数据集名称
            task_type: 任务类型
            sample: 数据样本
            max_context_tokens: 模型最大上下文token数

        Returns:
            实验结果字典
        """
        context = sample.get("context", "")
        question = sample.get("question", "")

        # 计算自然token长度
        prompt = self.build_prompt(task_type, context, question)
        total_tokens = self.count_tokens(prompt)
        natural_ratio = total_tokens / max_context_tokens if max_context_tokens > 0 else 0

        logger.info(f"样本 {sample.get('id', 'unknown')}: "
                   f"tokens={total_tokens}, ratio={natural_ratio:.2%}, "
                   f"model={model_name}, task={task_type}")

        # 调用模型（使用完整的原始文本，不截断）
        start_time = time.time()
        try:
            response = self.llm_client.generate(
                model=model_name,
                prompt=prompt,
                temperature=EXPERIMENT_CONFIG["temperature"],
                max_tokens=EXPERIMENT_CONFIG["max_tokens"]
            )
            elapsed_time = time.time() - start_time
            prediction = response.get("response", "")
        except Exception as e:
            logger.error(f"模型调用失败: {e}")
            elapsed_time = time.time() - start_time
            prediction = ""

        # 评估结果
        evaluator = get_evaluator(task_type)

        # 提取reference
        reference = ""
        if isinstance(sample.get("answers"), list) and len(sample["answers"]) > 0:
            if isinstance(sample["answers"][0], list) and len(sample["answers"][0]) > 0:
                reference = sample["answers"][0][0]
            elif isinstance(sample["answers"][0], str):
                reference = sample["answers"][0]
        elif isinstance(sample.get("answers"), str):
            reference = sample["answers"]

        # 计算评估指标
        if reference:
            metrics = evaluator.evaluate(prediction, reference, context=context)
        else:
            logger.warning(f"样本缺少reference: {sample.get('id', 'unknown')}")
            # 获取评估器的默认指标
            if hasattr(evaluator, 'get_metric_names'):
                metrics = {k: 0.0 for k in evaluator.get_metric_names()}
            else:
                # 根据任务类型提供默认指标
                if task_type == 'reading_comprehension':
                    metrics = {'f1': 0.0}
                else:
                    metrics = {}

        logger.info(f"实验完成: metrics={metrics}")

        # 返回结果
        result = {
            "sample_id": sample.get("id", "unknown"),
            "model": model_name,
            "model_key": model_key,
            "dataset": dataset_name,
            "task_type": task_type,
            "natural_tokens": total_tokens,
            "natural_ratio": natural_ratio,
            "max_context_tokens": max_context_tokens,
            "context_length": len(context),
            "question_length": len(question),
            "prediction": prediction,
            "reference": reference,
            "metrics": metrics,
            "elapsed_time": elapsed_time,
            "timestamp": datetime.now().isoformat()
        }

        return result

    def run_experiment(self,
                      model_keys: List[str],
                      dataset_name: str,
                      task_types: List[str],
                      max_samples: int = 100,
                      max_ratio: float = 0.95) -> None:
        """
        运行完整实验

        Args:
            model_keys: 模型键名列表
            dataset_name: 数据集名称
            task_types: 任务类型列表
            max_samples: 最大样本数
            max_ratio: 最大上下文比率（默认95%，过滤超长样本）
        """
        logger.info("="*60)
        logger.info("开始自然长度分布实验")
        logger.info("="*60)
        logger.info(f"实验模式: 使用样本自然长度（不截断/填充）")
        logger.info(f"数据集: {dataset_name}")
        logger.info(f"模型: {model_keys}")
        logger.info(f"任务类型: {task_types}")
        logger.info(f"最大样本数: {max_samples}")
        logger.info(f"最大比率过滤: {max_ratio:.0%}")

        # 加载数据集
        logger.info(f"\n加载数据集: {dataset_name}")
        dataset_loader = get_dataset_loader(dataset_name, max_samples=max_samples * 3)  # 多加载一些用于过滤
        all_samples = dataset_loader.load()
        logger.info(f"初始加载了 {len(all_samples)} 个样本")

        # 遍历所有配置
        total_configs = len(model_keys) * len(task_types)
        current_config = 0

        for model_key in model_keys:
            model_info = QWEN_MODELS[model_key]
            model_name = self.get_model_name(model_info, model_key)
            max_context_tokens = get_model_max_context(model_key)

            logger.info(f"\n测试模型: {model_key} ({model_name}) 后端={self.llm_backend}")
            logger.info(f"最大上下文: {max_context_tokens} tokens")

            # 过滤样本：只保留自然长度在合理范围内的样本
            # 注意：这里使用第一个任务类型来估算token数，因为不同任务类型的prompt模板可能略有不同
            # 但主要token数来自context和question，所以这个估算应该是合理的
            logger.info(f"\n过滤样本（保留比率 < {max_ratio:.0%}）...")
            filtered_samples = []
            # 使用第一个任务类型来构建prompt以准确计算token数
            first_task_type = task_types[0] if task_types else 'reading_comprehension'
            for sample in all_samples:
                context = sample.get("context", "")
                question = sample.get("question", "")
                # 使用精确的token计算（与运行时一致）
                # 构建prompt以准确计算token数（使用第一个任务类型）
                prompt = self.build_prompt(first_task_type, context, question)
                estimated_tokens = self.count_tokens(prompt)  # 使用tiktoken精确计算
                ratio = estimated_tokens / max_context_tokens if max_context_tokens > 0 else 0

                if ratio < max_ratio:
                    filtered_samples.append(sample)
                    if len(filtered_samples) >= max_samples:
                        break

            samples = filtered_samples[:max_samples]
            logger.info(f"过滤后保留 {len(samples)} 个样本")

            if len(samples) == 0:
                logger.warning(f"没有合适的样本，跳过模型 {model_key}")
                continue

            # 检查模型可用性
            if not self.llm_client.check_model_available(model_name):
                logger.error(f"模型 {model_name} 不可用，跳过")
                continue
            
            for task_type in task_types:
                current_config += 1
                logger.info(f"\n进度: [{current_config}/{total_configs}] "
                           f"模型={model_key}, 任务={task_type}")
                
                # 运行所有样本
                results = []
                for idx, sample in enumerate(samples, 1):
                    logger.info(f"\n样本 {idx}/{len(samples)}")
                    
                    result = self.run_single_sample(
                        model_name=model_name,
                        model_key=model_key,
                        dataset_name=dataset_name,
                        task_type=task_type,
                        sample=sample,
                        max_context_tokens=max_context_tokens
                    )
                    
                    results.append(result)
                
                # 保存结果
                self.save_results(results, model_key, dataset_name, task_type)
        
        logger.info("\n" + "="*60)
        logger.info("实验完成！")
        logger.info("="*60)
    
    def save_results(self, results: List[Dict], model_key: str, 
                    dataset_name: str, task_type: str) -> None:
        """
        保存实验结果
        
        Args:
            results: 结果列表
            model_key: 模型键名
            dataset_name: 数据集名称
            task_type: 任务类型
        """
        # 生成文件名
        filename = f"natural_length_{model_key}_{dataset_name}_{task_type}.json"
        filepath = self.dataset_dir / filename
        
        # 准备保存的数据
        save_data = {
            "metadata": {
                "experiment_type": "natural_length",
                "model": model_key,
                "dataset": dataset_name,
                "task_type": task_type,
                "num_samples": len(results),
                "timestamp": datetime.now().isoformat()
            },
            "results": results
        }
        
        # 保存JSON文件
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ 结果已保存: {filepath}")
        
        # 打印统计信息
        self.print_statistics(results)
    
    def print_statistics(self, results: List[Dict]) -> None:
        """
        打印统计信息
        
        Args:
            results: 结果列表
        """
        if not results:
            return
        
        # 提取数据
        ratios = [r['natural_ratio'] for r in results]
        
        # 获取第一个结果的metric名称
        metric_names = list(results[0]['metrics'].keys())
        
        logger.info("\n" + "="*60)
        logger.info("统计信息")
        logger.info("="*60)
        
        # 上下文长度分布
        logger.info(f"\n【上下文长度分布】")
        logger.info(f"  样本数: {len(results)}")
        logger.info(f"  比率范围: {min(ratios):.2%} - {max(ratios):.2%}")
        logger.info(f"  平均比率: {sum(ratios)/len(ratios):.2%}")
        
        # 按比率区间统计
        bins = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), 
                (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
        
        logger.info(f"\n【按比率区间分布】")
        for bin_start, bin_end in bins:
            count = sum(1 for r in ratios if bin_start <= r < bin_end)
            if count > 0:
                logger.info(f"  {bin_start:.0%}-{bin_end:.0%}: {count} 个样本")
        
        # 性能指标统计
        logger.info(f"\n【性能指标】")
        for metric_name in metric_names:
            values = [r['metrics'][metric_name] for r in results]
            avg_value = sum(values) / len(values) if values else 0
            logger.info(f"  {metric_name}: 平均={avg_value:.4f}")
        
        logger.info("="*60)

