"""
数据集准备脚本
下载真实数据集并采样500条保存到data目录
"""

import json
import random
import logging
from pathlib import Path
from typing import List, Dict

from utils.dataset_loader import (
    get_dataset_loader
)

# 配置日志
# 确保logs目录存在
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# 配置日志：同时输出到控制台和文件
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(logs_dir / "prepare_datasets.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def prepare_squad(data_dir: Path, max_samples: int = 500, random_seed: int = 42):
    """
    准备SQuAD数据集
    
    Args:
        data_dir: 数据目录
        max_samples: 最大采样数量
        random_seed: 随机种子
    """
    logger.info("="*60)
    logger.info("准备 SQuAD 数据集")
    logger.info("="*60)
    
    dataset_path = data_dir / "squad.json"
    temp_path = data_dir / "squad_full.json"
    
    # 如果已有采样后的文件，先备份或删除
    if dataset_path.exists():
        logger.info(f"发现已有数据集文件: {dataset_path}")
        logger.info("将重新下载并采样...")
    
    # 直接从Hugging Face下载指定数量的数据（不下载全部）
    logger.info(f"正在从Hugging Face下载SQuAD数据集（只下载 {max_samples} 条，不下载全部数据）...")
    try:
        from datasets import load_dataset
        
        # 加载数据集
        logger.info("正在加载数据集...")
        dataset = load_dataset("squad", "plain_text", split="train")
        
        # 只下载指定数量的数据（使用select方法，更高效）
        logger.info(f"从数据集中采样 {max_samples} 条...")
        # 先获取数据集大小（这不会下载数据）
        dataset_size = len(dataset)
        logger.info(f"数据集总大小: {dataset_size} 条")
        
        # 生成随机索引（只取max_samples * 2个索引，确保有足够的数据）
        random.seed(random_seed)
        sample_size = min(max_samples * 2, dataset_size)
        random_indices = random.sample(range(dataset_size), sample_size)
        random.seed()
        
        # 只选择这些索引的数据（这样只会下载选中的数据）
        dataset = dataset.select(random_indices)
        logger.info(f"已选择 {len(dataset)} 条数据进行处理")
        
        all_data = []
        for item in dataset:
            all_data.append({
                "context": item["context"],
                "question": item["question"],
                "answers": item["answers"]["text"],
                "id": item["id"]
            })
            
            # 如果已经收集到足够的样本，提前停止
            if len(all_data) >= max_samples:
                break
        
        logger.info(f"下载并转换完成，共 {len(all_data)} 条数据")
        
        # 如果数据不足，再次采样
        if len(all_data) > max_samples:
            random.seed(random_seed)
            sampled_data = random.sample(all_data, max_samples)
            random.seed()
            logger.info(f"从 {len(all_data)} 条数据中采样了 {max_samples} 条")
        else:
            sampled_data = all_data
            logger.info(f"数据量少于 {max_samples} 条，使用全部 {len(all_data)} 条")
    except ImportError:
        logger.error("需要安装datasets库: pip install datasets")
        raise
    except Exception as e:
        logger.error(f"下载失败: {e}")
        raise
    
    # 保存采样后的数据
    with open(dataset_path, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"已保存 {len(sampled_data)} 条数据到: {dataset_path}")
    logger.info("")
    
    return sampled_data


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="准备实验数据集")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="数据目录（默认: data）"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="每个数据集的最大采样数量（默认: 500）"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="随机种子（默认: 42）"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["squad", "narrativeqa", "triviaqa", "all"],
        default=["all"],
        help="要准备的数据集（默认: all）。推荐使用narrativeqa或triviaqa作为长上下文数据集"
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    data_dir.mkdir(exist_ok=True)
    
    logger.info("="*60)
    logger.info("开始准备数据集")
    logger.info("="*60)
    logger.info(f"数据目录: {data_dir}")
    logger.info(f"采样数量: {args.max_samples} 条/数据集")
    logger.info(f"随机种子: {args.random_seed}")
    logger.info("")
    
    datasets_to_prepare = []
    if "all" in args.datasets:
        # 默认只准备squad（向后兼容），如需长上下文数据集请明确指定
        datasets_to_prepare = ["squad"]
    else:
        datasets_to_prepare = args.datasets
    
    results = {}
    
    # 准备数据集（使用统一的数据加载器）
    for dataset_name in datasets_to_prepare:
        try:
            logger.info("="*60)
            logger.info(f"准备 {dataset_name.upper()} 数据集")
            logger.info("="*60)
            
            loader = get_dataset_loader(
                dataset_name,
                data_dir=str(data_dir),
                max_samples=args.max_samples,
                random_seed=args.random_seed
            )
            
            # 加载数据（会自动下载如果不存在）
            data = loader.load()
            results[dataset_name] = data
            
            logger.info(f"✓ {dataset_name}: 成功加载 {len(data)} 条数据")
            if data:
                avg_len = sum(len(item.get("context", "")) for item in data) / len(data)
                logger.info(f"  平均上下文长度: {avg_len:.0f} 字符 (约 {avg_len/4:.0f} tokens)")
            logger.info("")
            
        except Exception as e:
            logger.error(f"✗ {dataset_name}: 准备失败 - {e}")
            results[dataset_name] = None
            logger.info("")
    
    # 总结
    logger.info("="*60)
    logger.info("数据集准备完成")
    logger.info("="*60)
    
    for dataset_name, data in results.items():
        if data is not None:
            logger.info(f"✓ {dataset_name}: {len(data)} 条数据")
        else:
            logger.info(f"✗ {dataset_name}: 准备失败")
    
    logger.info("")
    logger.info("现在可以运行实验了:")
    logger.info("  python main.py run-all")
    logger.info("")


if __name__ == "__main__":
    main()

