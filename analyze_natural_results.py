"""
分析自然长度实验结果
绘制散点图：自然比率 vs. 性能指标
检测断崖点
"""

import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from scipy import stats
from scipy.interpolate import UnivariateSpline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    HAS_MATPLOTLIB = True
except ImportError:
    logger.warning("matplotlib未安装，无法生成图表。请运行: pip install matplotlib")
    HAS_MATPLOTLIB = False


class NaturalLengthAnalyzer:
    """自然长度实验结果分析器"""
    
    def __init__(self, results_dir: str, dataset_name: str):
        """
        初始化分析器
        
        Args:
            results_dir: 结果目录
            dataset_name: 数据集名称
        """
        self.results_dir = Path(results_dir) / dataset_name
        self.dataset_name = dataset_name
        self.plots_dir = Path("plots") / dataset_name / "natural_length"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
    
    def load_results(self, model_key: str, task_type: str) -> Dict:
        """
        加载实验结果
        
        Args:
            model_key: 模型键名
            task_type: 任务类型
            
        Returns:
            结果字典
        """
        filename = f"natural_length_{model_key}_{self.dataset_name}_{task_type}.json"
        filepath = self.results_dir / filename
        
        if not filepath.exists():
            logger.error(f"结果文件不存在: {filepath}")
            return None
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"加载结果: {filename}")
        return data
    
    def extract_data(self, results: List[Dict], metric_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        提取数据点
        
        Args:
            results: 结果列表
            metric_name: 指标名称
            
        Returns:
            (ratios, values) - numpy数组
        """
        ratios = []
        values = []
        
        for result in results:
            ratio = result.get('natural_ratio', 0)
            metric_value = result.get('metrics', {}).get(metric_name, 0)
            
            ratios.append(ratio)
            values.append(metric_value)
        
        return np.array(ratios), np.array(values)
    
    def detect_cliff_point(self, ratios: np.ndarray, values: np.ndarray, 
                          threshold: float = 0.3) -> Dict:
        """
        检测断崖点
        
        Args:
            ratios: 比率数组
            values: 性能值数组
            threshold: 降级阈值（默认30%）
            
        Returns:
            检测结果字典
        """
        if len(ratios) < 10:
            logger.warning("样本数太少，无法可靠检测断崖点")
            return {"detected": False}
        
        # 按ratio排序
        sorted_indices = np.argsort(ratios)
        sorted_ratios = ratios[sorted_indices]
        sorted_values = values[sorted_indices]
        
        # 方法1: 梯度分析 - 找最大下降点
        if len(sorted_values) > 1:
            # 计算梯度，避免除以零
            ratio_diffs = np.diff(sorted_ratios)
            ratio_diffs[ratio_diffs == 0] = 1e-10  # 避免除以零
            gradients = np.diff(sorted_values) / ratio_diffs
            
            # 找到最陡下降点（负梯度最大）
            max_drop_idx = np.argmin(gradients)
            
            # 计算基线性能（前20%的平均值）
            baseline_size = max(5, len(sorted_values) // 5)
            baseline_value = np.mean(sorted_values[:baseline_size])
            
            # 断崖点之后的性能
            after_cliff_size = min(10, len(sorted_values) - max_drop_idx - 1)
            if after_cliff_size > 0:
                drop_value = np.mean(sorted_values[max_drop_idx + 1:max_drop_idx + 1 + after_cliff_size])
            else:
                drop_value = sorted_values[max_drop_idx + 1] if max_drop_idx + 1 < len(sorted_values) else sorted_values[-1]
            
            # 计算性能下降百分比
            if baseline_value > 0:
                drop_percentage = (baseline_value - drop_value) / baseline_value
            else:
                drop_percentage = 0
            
            cliff_ratio = sorted_ratios[max_drop_idx]
            
            logger.info(f"\n【断崖点检测】")
            logger.info(f"  检测到的临界比率: {cliff_ratio:.2%}")
            logger.info(f"  基线性能: {baseline_value:.4f}")
            logger.info(f"  断崖后性能: {drop_value:.4f}")
            logger.info(f"  性能下降: {drop_percentage:.1%}")
            
            is_cliff = drop_percentage >= threshold
            
            if is_cliff:
                logger.info(f"  ✅ 检测到断崖式降智 (下降 {drop_percentage:.1%} ≥ {threshold:.0%})")
            else:
                logger.info(f"  🟡 性能下降较缓和 (下降 {drop_percentage:.1%} < {threshold:.0%})")
            
            return {
                "detected": is_cliff,
                "cliff_ratio": float(cliff_ratio),
                "baseline_performance": float(baseline_value),
                "cliff_performance": float(drop_value),
                "drop_percentage": float(drop_percentage)
            }
        
        return {"detected": False}
    
    def plot_scatter(self, ratios: np.ndarray, values: np.ndarray,
                    model_key: str, task_type: str, metric_name: str,
                    cliff_info: Dict = None) -> None:
        """
        绘制散点图
        
        Args:
            ratios: 比率数组
            values: 性能值数组
            model_key: 模型键名
            task_type: 任务类型
            metric_name: 指标名称
            cliff_info: 断崖点信息
        """
        if not HAS_MATPLOTLIB:
            logger.warning("matplotlib未安装，跳过绘图")
            return
        
        plt.figure(figsize=(12, 7))
        
        # 绘制散点
        plt.scatter(ratios * 100, values, alpha=0.6, s=50, label='样本数据点')
        
        # 添加趋势线
        if len(ratios) > 10:
            sorted_indices = np.argsort(ratios)
            sorted_ratios = ratios[sorted_indices]
            sorted_values = values[sorted_indices]
            
            # 使用滑动窗口平滑
            window_size = max(5, len(sorted_ratios) // 10)
            if window_size % 2 == 0:  # 确保窗口大小是奇数
                window_size += 1
            
            # 使用mode='same'保持长度一致
            smoothed_values = np.convolve(sorted_values, 
                                         np.ones(window_size)/window_size, 
                                         mode='same')
            
            plt.plot(sorted_ratios * 100, smoothed_values, 
                    'r-', linewidth=2, label='趋势线（滑动平均）', alpha=0.8)
        
        # 标记断崖点
        if cliff_info and cliff_info.get('detected'):
            cliff_ratio = cliff_info['cliff_ratio']
            plt.axvline(x=cliff_ratio * 100, color='red', linestyle='--', 
                       linewidth=2, label=f'断崖点 ({cliff_ratio:.1%})')
            
            # 添加注释
            plt.text(cliff_ratio * 100 + 2, plt.ylim()[1] * 0.9,
                    f'临界点: {cliff_ratio:.1%}\n'
                    f'性能下降: {cliff_info["drop_percentage"]:.1%}',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                    fontsize=10)
        
        plt.xlabel('上下文长度比率 (%)', fontsize=12)
        plt.ylabel(f'{metric_name}', fontsize=12)
        plt.title(f'{model_key} - {task_type} - {metric_name}\n'
                 f'自然长度分布分析', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        # 强制X轴显示到100%
        plt.xlim(0, 100)
        
        # 保存图片
        filename = f"scatter_{model_key}_{task_type}_{metric_name}.png"
        filepath = self.plots_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ 散点图已保存: {filepath}")
    
    def analyze_results(self, model_key: str, task_type: str) -> None:
        """
        分析指定模型和任务的结果
        
        Args:
            model_key: 模型键名
            task_type: 任务类型
        """
        logger.info("\n" + "="*60)
        logger.info(f"分析: {model_key} - {task_type}")
        logger.info("="*60)
        
        # 加载结果
        data = self.load_results(model_key, task_type)
        if not data:
            return
        
        results = data.get('results', [])
        if not results:
            logger.warning("结果为空")
            return
        
        logger.info(f"样本数: {len(results)}")
        
        # 获取所有指标名称
        metric_names = list(results[0].get('metrics', {}).keys())
        logger.info(f"指标: {metric_names}")
        
        # 分析每个指标
        for metric_name in metric_names:
            logger.info(f"\n分析指标: {metric_name}")
            
            # 提取数据
            ratios, values = self.extract_data(results, metric_name)
            
            if len(ratios) == 0:
                logger.warning(f"指标 {metric_name} 无数据")
                continue
            
            # 打印统计信息
            logger.info(f"  比率范围: {ratios.min():.2%} - {ratios.max():.2%}")
            logger.info(f"  {metric_name} 范围: {values.min():.4f} - {values.max():.4f}")
            logger.info(f"  {metric_name} 平均: {values.mean():.4f}")
            
            # 检测断崖点
            cliff_info = self.detect_cliff_point(ratios, values)
            
            # 绘制散点图
            self.plot_scatter(ratios, values, model_key, task_type, 
                            metric_name, cliff_info)
    
    def analyze_all_results(self, model_key_filter: str = None, task_type_filter: str = None) -> None:
        """
        分析所有结果文件
        
        Args:
            model_key_filter: 可选的模型过滤（只分析指定模型）
            task_type_filter: 可选的任务类型过滤（只分析指定任务）
        """
        result_files = list(self.results_dir.glob("natural_length_*.json"))
        
        if not result_files:
            logger.error(f"未找到结果文件: {self.results_dir}")
            return
        
        logger.info(f"找到 {len(result_files)} 个结果文件")
        
        for result_file in result_files:
            # 解析文件名: natural_length_{model}_{dataset}_{task}.json
            # 例如: natural_length_qwen2.5-7b_mixed_reading_comprehension.json.bak
            filename = result_file.stem  # 去掉.json后缀
            
            # 移除前缀 "natural_length_"
            if filename.startswith("natural_length_"):
                filename = filename[len("natural_length_"):]
            
            # 分割剩余部分: qwen2.5-7b_mixed_reading_comprehension
            parts = filename.split('_')
            
            if len(parts) >= 3:
                # parts[0] = model_key (如 qwen2.5-7b)
                # parts[1] = dataset (如 mixed)
                # parts[2:] = task_type (如 reading, comprehension -> reading_comprehension)
                model_key = parts[0]
                dataset = parts[1]
                task_type = '_'.join(parts[2:])  # 重新组合任务类型
                
                # 应用过滤
                if model_key_filter and model_key != model_key_filter:
                    continue
                if task_type_filter and task_type != task_type_filter:
                    continue
                
                logger.info(f"解析文件: model={model_key}, dataset={dataset}, task={task_type}")
                self.analyze_results(model_key, task_type)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="分析自然长度实验结果",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--dataset', type=str, required=True,
                       help='数据集名称')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='结果目录（默认: results）')
    parser.add_argument('--model', type=str,
                       help='指定模型（可选），不指定则分析所有模型')
    parser.add_argument('--task', type=str,
                       help='指定任务类型（可选），如 reading_comprehension, summarization。不指定则分析所有任务')
    
    args = parser.parse_args()
    
    analyzer = NaturalLengthAnalyzer(args.results_dir, args.dataset)
    
    if args.model and args.task:
        # 分析指定的模型和任务
        analyzer.analyze_results(args.model, args.task)
    else:
        # 分析所有结果（应用过滤）
        analyzer.analyze_all_results(
            model_key_filter=args.model,
            task_type_filter=args.task
        )
    
    logger.info("\n" + "="*60)
    logger.info("✅ 分析完成！")
    logger.info(f"图表已保存到: {analyzer.plots_dir}")
    logger.info("="*60)


if __name__ == '__main__':
    main()

