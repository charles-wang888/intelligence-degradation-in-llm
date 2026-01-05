"""
基于自然长度分布的实验主入口
使用每个样本的自然token长度，不进行任何截断或填充
"""

import argparse
import sys
import logging
from pathlib import Path

from config import QWEN_MODELS, TASK_TYPES, DATASETS, OLLAMA_CONFIG, VLLM_CONFIG
from experiment_natural import NaturalLengthExperimentRunner

# 配置日志
Path('logs').mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="大模型降智现象实验 - 自然长度分布分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

  # 测试单个模型的单个任务
  python main_natural.py --dataset narrativeqa --model qwen2.5-3b --task reading_comprehension --max-samples 100
  
  # 测试单个模型的所有任务
  python main_natural.py --dataset narrativeqa --model qwen2.5-7b --max-samples 100
  
  # 测试所有模型的所有任务
  python main_natural.py --dataset narrativeqa --max-samples 100
  
  # 快速测试（20个样本）
  python main_natural.py --dataset narrativeqa --model qwen2.5-3b --task reading_comprehension --max-samples 20
  
  # 使用更严格的过滤（只保留比率<80%的样本）
  python main_natural.py --dataset narrativeqa --model qwen2.5-3b --task reading_comprehension --max-samples 100 --max-ratio 0.8
  
  # 使用vLLM后端（示例地址与密钥）
  python main_natural.py --dataset mixed --model qwen2.5-7b --task reading_comprehension --max-samples 1000 \\
    --llm-backend vllm --vllm-url http://10.22.0.120:11434/v1 --vllm-api-key 7ttyjVwJDChjSC67

实验原理:
  - 使用每个样本的自然token长度（不截断/填充）
  - 计算每个样本的 token数/最大上下文 比率
  - 记录：(自然比率, F1分数, 其他metrics)
  - 分析：自然比率 vs. 性能的关系，找出断崖点
        """
    )

    parser.add_argument('--dataset', type=str, required=True,
                       choices=list(DATASETS.keys()),
                       help=f'数据集名称（必需）。可选值: {list(DATASETS.keys())}')

    parser.add_argument('--model', type=str,
                       choices=list(QWEN_MODELS.keys()),
                       help=f'指定单个模型（可选），如不指定则运行所有模型。可选值: {list(QWEN_MODELS.keys())}')

    parser.add_argument('--task', type=str,
                       choices=list(TASK_TYPES.keys()),
                       help=f'指定任务类型（可选），如不指定则运行所有任务。可选值: {list(TASK_TYPES.keys())}')

    parser.add_argument('--max-samples', type=int, default=100,
                       help='最大样本数（默认: 100）')

    parser.add_argument('--max-ratio', type=float, default=0.95,
                       help='最大上下文比率，过滤超长样本（默认: 0.95）')

    parser.add_argument('--results-dir', type=str, default='results',
                       help='结果保存目录（默认: results）')

    parser.add_argument('--llm-backend', type=str, choices=['ollama', 'vllm'], default='ollama',
                       help='LLM后端，支持ollama或vllm（默认: ollama）')

    parser.add_argument('--ollama-url', type=str, default=OLLAMA_CONFIG["base_url"],
                       help=f'Ollama服务地址（默认: {OLLAMA_CONFIG["base_url"]}）')

    parser.add_argument('--vllm-url', type=str, default=VLLM_CONFIG["base_url"],
                       help=f'vLLM服务地址（OpenAI兼容，默认: {VLLM_CONFIG["base_url"]}）')

    parser.add_argument('--vllm-api-key', type=str, default=VLLM_CONFIG.get("api_key"),
                       help='vLLM API key，如未设置可留空')

    args = parser.parse_args()

    # 创建必要的目录
    Path('logs').mkdir(exist_ok=True)
    Path('results').mkdir(exist_ok=True)
    Path('data').mkdir(exist_ok=True)

    logger.info("="*60)
    logger.info("自然长度分布实验")
    logger.info("="*60)
    logger.info("实验模式: 使用样本自然长度（不截断/填充）")
    logger.info(f"数据集: {args.dataset}")
    logger.info(f"LLM后端: {args.llm_backend}")
    if args.llm_backend == "vllm":
        logger.info(f"vLLM地址: {args.vllm_url}")
    else:
        logger.info(f"Ollama地址: {args.ollama_url}")

    # 确定模型列表
    if args.model:
        model_keys = [args.model]
        logger.info(f"指定模型: {args.model}")
    else:
        model_keys = list(QWEN_MODELS.keys())
        logger.info(f"运行所有模型: {model_keys}")

    # 确定任务列表
    if args.task:
        task_types = [args.task]
        logger.info(f"指定任务: {args.task}")
    else:
        # 使用数据集支持的任务
        dataset_config = DATASETS.get(args.dataset, {})
        task_types = dataset_config.get("supported_tasks", list(TASK_TYPES.keys()))
        logger.info(f"运行任务: {task_types}")

    logger.info(f"最大样本数: {args.max_samples}")
    logger.info(f"总配置数: {len(model_keys)} 模型 × {len(task_types)} 任务 = {len(model_keys) * len(task_types)}")
    logger.info("="*60)

    # 创建实验运行器
    runner = NaturalLengthExperimentRunner(
        results_dir=args.results_dir,
        ollama_url=args.ollama_url,
        vllm_url=args.vllm_url,
        vllm_api_key=args.vllm_api_key,
        llm_backend=args.llm_backend,
        dataset_name=args.dataset
    )
    
    # 运行实验
    runner.run_experiment(
        model_keys=model_keys,
        dataset_name=args.dataset,
        task_types=task_types,
        max_samples=args.max_samples,
        max_ratio=args.max_ratio
    )
    
    logger.info("\n" + "="*60)
    logger.info("✅ 所有实验完成！")
    logger.info("="*60)
    logger.info(f"\n结果已保存到: {Path(args.results_dir) / args.dataset}")
    logger.info("\n下一步: 运行分析脚本可视化结果")
    logger.info("  python analyze_natural_results.py --dataset " + args.dataset)


if __name__ == '__main__':
    main()

