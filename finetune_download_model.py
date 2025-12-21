"""
下载Qwen2.5-7B模型到本地
"""

import os
import logging
from pathlib import Path
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_model(model_name: str, local_dir: str, cache_dir: Optional[str] = None):
    """
    下载模型到本地
    
    Args:
        model_name: HuggingFace模型名称
        local_dir: 本地保存目录
        cache_dir: HuggingFace缓存目录（可选）
    """
    logger.info("="*60)
    logger.info(f"下载模型: {model_name}")
    logger.info(f"保存到: {local_dir}")
    logger.info("="*60)
    
    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)
    
    # 设置环境变量（如果需要）
    if cache_dir:
        os.environ["HF_HOME"] = cache_dir
        os.environ["TRANSFORMERS_CACHE"] = cache_dir
    
    try:
        logger.info("正在下载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        tokenizer.save_pretrained(local_dir)
        logger.info("✓ Tokenizer下载完成")
        
        logger.info("正在下载模型（这可能需要较长时间，7B模型约14GB）...")
        logger.info("提示：如果下载中断，可以重新运行此脚本，会自动续传")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
            torch_dtype="auto",  # 自动选择dtype
            low_cpu_mem_usage=True
        )
        
        logger.info("正在保存模型...")
        model.save_pretrained(local_dir)
        logger.info("✓ 模型下载完成")
        
        logger.info("="*60)
        logger.info("下载完成！")
        logger.info(f"模型已保存到: {local_dir}")
        logger.info("="*60)
        
        # 显示模型信息
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"模型参数量: {total_params / 1e9:.2f}B")
        
        return True
        
    except Exception as e:
        logger.error(f"下载失败: {e}")
        logger.error("请检查网络连接和磁盘空间")
        return False


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="下载Qwen2.5-7B模型")
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace模型名称"
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default="models/qwen2.5-7b-instruct",
        help="本地保存目录"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="HuggingFace缓存目录（可选）"
    )
    
    args = parser.parse_args()
    
    success = download_model(
        args.model_name,
        args.local_dir,
        args.cache_dir
    )
    
    if success:
        logger.info("\n现在可以使用本地模型进行微调：")
        logger.info(f"python finetune_cliff_mitigation.py --model {args.local_dir}")
    else:
        logger.error("\n下载失败，请检查错误信息")


if __name__ == "__main__":
    main()

