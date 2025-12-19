"""
精确检测LLM断崖式降智点
基于results/mixed目录下的JSON文件，使用多种方法精确找到断崖点位置
"""

import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CliffPointDetector:
    """断崖点检测器 - 使用多种方法精确检测"""
    
    def __init__(self, results_file: Path):
        """
        初始化检测器
        
        Args:
            results_file: 结果JSON文件路径
        """
        self.results_file = results_file
        self.data = None
        self.ratios = None
        self.values = None
        self.sorted_indices = None
        self.sorted_ratios = None
        self.sorted_values = None
        
    def load_data(self) -> bool:
        """加载数据"""
        try:
            with open(self.results_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            
            results = self.data.get('results', [])
            if not results:
                logger.error("结果数据为空")
                return False
            
            logger.info(f"✅ 成功加载 {len(results)} 个样本")
            return True
        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            return False
    
    def extract_metric_data(self, metric_name: str = 'f1') -> bool:
        """
        提取指标数据
        
        过滤规则：
        - 只过滤掉 F1=0 和 F1=1 的极端值数据点
        - 不指定比率范围，使用所有有效数据
        
        Args:
            metric_name: 指标名称（默认'f1'）
        """
        if not self.data:
            logger.error("请先加载数据")
            return False
        
        ratios = []
        values = []
        
        for result in self.data.get('results', []):
            ratio = result.get('natural_ratio', 0)
            metric_value = result.get('metrics', {}).get(metric_name, 0)
            
            # 过滤条件：
            # 1. ratio > 0（有效数据）
            # 2. 过滤极端值：metric_value != 0 且 metric_value != 1（排除F1=0和F1=1的极端情况）
            if ratio > 0 and metric_value != 0.0 and metric_value != 1.0:
                ratios.append(ratio)
                values.append(metric_value)
        
        if len(ratios) == 0:
            logger.error(f"未找到有效的{metric_name}数据（已过滤F1=0和F1=1的极端值）")
            return False
        
        self.ratios = np.array(ratios)
        self.values = np.array(values)
        
        # 按ratio排序
        self.sorted_indices = np.argsort(self.ratios)
        self.sorted_ratios = self.ratios[self.sorted_indices]
        self.sorted_values = self.values[self.sorted_indices]
        
        logger.info(f"✅ 提取了 {len(self.ratios)} 个有效数据点（已过滤F1=0和F1=1的极端值）")
        logger.info(f"   比率范围: {self.sorted_ratios[0]:.2%} - {self.sorted_ratios[-1]:.2%}")
        logger.info(f"   {metric_name} 范围: {self.sorted_values.min():.4f} - {self.sorted_values.max():.4f}")
        
        return True
    
    def find_all_peaks(self, min_ratio: float = 0.30, max_ratio: float = 0.60, 
                      min_peak_height: float = 0.3, window_size: int = 5) -> List[Dict]:
        """
        找到所有局部峰值点
        
        Args:
            min_ratio: 最小比率范围
            max_ratio: 最大比率范围
            min_peak_height: 最小峰值高度（F1值）
            window_size: 峰值检测窗口大小
            
        Returns:
            峰值点列表，每个元素包含 {ratio, value, index, sustained_decline}
        """
        # 在指定范围内查找峰值
        region_mask = (self.sorted_ratios >= min_ratio) & (self.sorted_ratios <= max_ratio)
        if not np.any(region_mask):
            return []
        
        region_indices = np.where(region_mask)[0]
        region_ratios = self.sorted_ratios[region_indices]
        region_values = self.sorted_values[region_indices]
        
        peaks = []
        
        # 使用滑动窗口找局部峰值
        for i in range(window_size, len(region_values) - window_size):
            center_idx = region_indices[i]
            center_value = region_values[i]
            
            # 检查是否是局部峰值（比左右窗口内的值都大）
            left_window = region_values[i - window_size:i]
            right_window = region_values[i + 1:i + window_size + 1]
            
            if len(left_window) > 0 and len(right_window) > 0:
                if center_value >= np.max(left_window) and center_value >= np.max(right_window):
                    # 检查峰值高度
                    if center_value >= min_peak_height:
                        peak_ratio = region_ratios[i]
                        
                        # 关键检查：峰值点往后10%的比率范围内是否存在上升情况
                        target_ratio = peak_ratio + 0.10  # 往后10%
                        
                        # 找到目标比率范围内的数据点
                        target_mask = (self.sorted_ratios > peak_ratio) & (self.sorted_ratios <= target_ratio)
                        target_indices = np.where(target_mask)[0]
                        
                        # 如果目标范围内有数据点，检查是否存在上升
                        if len(target_indices) > 0:
                            target_values = self.sorted_values[target_indices]
                            
                            # 检查1：是否有任何值超过峰值
                            max_in_range = np.max(target_values)
                            exceeds_peak = max_in_range > center_value
                            
                            # 检查2：是否有明显的上升趋势（连续上升）
                            has_rising_trend = False
                            max_consecutive_rises = 0
                            if len(target_values) > 2:
                                diffs = np.diff(target_values)
                                consecutive_rises = 0
                                for diff in diffs:
                                    if diff > 0:
                                        consecutive_rises += 1
                                        max_consecutive_rises = max(max_consecutive_rises, consecutive_rises)
                                    else:
                                        consecutive_rises = 0
                                has_rising_trend = max_consecutive_rises >= 3  # 连续3个点上升
                            
                            # 检查3：整体趋势是否上升（线性回归斜率）
                            is_rising_trend = False
                            slope = 0.0
                            if len(target_values) > 1:
                                target_indices_local = np.arange(len(target_values))
                                slope = np.polyfit(target_indices_local, target_values, 1)[0]
                                is_rising_trend = slope > 0  # 正斜率表示上升趋势
                            
                            # 如果往后10%范围内存在上升情况，排除这个候选点
                            has_rise_in_range = exceeds_peak or has_rising_trend or is_rising_trend
                            
                            if has_rise_in_range:
                                # 记录排除原因（用于日志输出）
                                reasons = []
                                if exceeds_peak:
                                    reasons.append(f"有值超过峰值(峰值={center_value:.4f}, 最大值={max_in_range:.4f})")
                                if has_rising_trend:
                                    reasons.append(f"连续{max_consecutive_rises}个点上升")
                                if is_rising_trend:
                                    reasons.append(f"整体趋势上升(斜率={slope:.4f})")
                                
                                # 这个候选点被排除，不添加到peaks列表
                                # 注意：这里不输出日志，因为会在调用方法中统一输出
                                continue
                        
                        # 如果通过了上升检查，继续分析持续下降
                        # 看峰值后至少30个点，或者到数据末尾
                        post_peak_end = min(center_idx + 1 + 50, len(self.sorted_values))
                        post_peak_values = self.sorted_values[center_idx + 1:post_peak_end]
                        
                        if len(post_peak_values) >= 10:  # 至少需要10个后续点才能判断
                            # 方法1：检查后续是否有明显的反弹（超过峰值的90%）
                            max_rebound = np.max(post_peak_values)
                            rebound_ratio = max_rebound / center_value if center_value > 0 else 1.0
                            
                            # 方法2：检查后续趋势是否整体下降（使用线性回归斜率）
                            post_peak_indices = np.arange(len(post_peak_values))
                            if len(post_peak_values) > 1:
                                # 计算线性趋势（斜率）
                                slope = np.polyfit(post_peak_indices, post_peak_values, 1)[0]
                                # 负斜率表示下降趋势
                                is_declining_trend = slope < 0
                            else:
                                is_declining_trend = False
                            
                            # 方法3：检查后续是否有连续上升段（反弹）
                            # 计算后续点的差分，看是否有连续的正差分（上升）
                            if len(post_peak_values) > 3:
                                diffs = np.diff(post_peak_values)
                                # 检查是否有连续3个以上的正差分（上升段）
                                consecutive_rises = 0
                                max_consecutive_rises = 0
                                for diff in diffs:
                                    if diff > 0:
                                        consecutive_rises += 1
                                        max_consecutive_rises = max(max_consecutive_rises, consecutive_rises)
                                    else:
                                        consecutive_rises = 0
                                has_significant_rebound = max_consecutive_rises >= 3  # 连续3个点上升
                            else:
                                has_significant_rebound = False
                            
                            # 综合判断：持续下降需要满足：
                            # 1. 最大反弹不超过峰值的85%（更严格）
                            # 2. 整体趋势下降（负斜率）
                            # 3. 没有明显的连续上升段
                            sustained_decline = (
                                rebound_ratio < 0.85 and  # 最大反弹不超过85%
                                is_declining_trend and    # 整体趋势下降
                                not has_significant_rebound  # 没有明显的连续上升
                            )
                            
                            # 计算下降幅度（使用后续30%的数据点）
                            lookback_size = min(30, len(post_peak_values))
                            post_peak_mean = np.mean(post_peak_values[:lookback_size])
                            drop_percentage = (center_value - post_peak_mean) / center_value if center_value > 0 else 0
                            
                            peaks.append({
                                "ratio": float(region_ratios[i]),
                                "ratio_percent": float(region_ratios[i] * 100),
                                "value": float(center_value),
                                "index": int(center_idx),
                                "sustained_decline": bool(sustained_decline),
                                "drop_percentage": float(drop_percentage),
                                "post_peak_mean": float(post_peak_mean),
                                "max_rebound": float(max_rebound),
                                "rebound_ratio": float(rebound_ratio),
                                "slope": float(slope) if len(post_peak_values) > 1 else 0.0,
                                "has_significant_rebound": bool(has_significant_rebound)
                            })
        
        # 按比率排序
        peaks.sort(key=lambda x: x["ratio"])
        
        # 返回峰值列表和排除信息（用于日志）
        return peaks
    
    def method1_gradient_analysis(self) -> Dict:
        """
        方法1: 梯度分析 - 找所有峰值点，选择持续下降的峰值作为断崖点
        
        策略：找到所有局部峰值，选择"峰值后持续下降且不反弹"的峰值作为断崖点
        
        Returns:
            检测结果字典，包含所有候选峰值点
        """
        logger.info("\n【方法1: 梯度分析（多峰值检测）】")
        
        if len(self.sorted_values) < 10:
            return {"method": "gradient", "detected": False, "reason": "样本数太少"}
        
        # 找到所有峰值点（在30%-60%范围内）
        # 先找到所有可能的峰值（包括被排除的）
        logger.info("   正在搜索峰值点（30%-60%范围）...")
        
        # 手动搜索峰值点，记录排除过程
        region_mask = (self.sorted_ratios >= 0.30) & (self.sorted_ratios <= 0.60)
        if not np.any(region_mask):
            logger.info("   未找到30%-60%范围内的数据")
            return {"method": "gradient", "detected": False, "reason": "未找到峰值点"}
        
        region_indices = np.where(region_mask)[0]
        region_ratios = self.sorted_ratios[region_indices]
        region_values = self.sorted_values[region_indices]
        
        found_peaks = []
        excluded_peaks = []
        window_size = 5
        min_peak_height = 0.3
        
        for i in range(window_size, len(region_values) - window_size):
            center_idx = region_indices[i]
            center_value = region_values[i]
            peak_ratio = region_ratios[i]
            
            left_window = region_values[i - window_size:i]
            right_window = region_values[i + 1:i + window_size + 1]
            
            if len(left_window) > 0 and len(right_window) > 0:
                if center_value >= np.max(left_window) and center_value >= np.max(right_window):
                    if center_value >= min_peak_height:
                        # 检查往后10%范围内是否有上升
                        target_ratio = peak_ratio + 0.10
                        target_mask = (self.sorted_ratios > peak_ratio) & (self.sorted_ratios <= target_ratio)
                        target_indices = np.where(target_mask)[0]
                        
                        if len(target_indices) > 0:
                            target_values = self.sorted_values[target_indices]
                            max_in_range = np.max(target_values)
                            exceeds_peak = max_in_range > center_value
                            
                            has_rising_trend = False
                            max_consecutive_rises = 0
                            if len(target_values) > 2:
                                diffs = np.diff(target_values)
                                consecutive_rises = 0
                                for diff in diffs:
                                    if diff > 0:
                                        consecutive_rises += 1
                                        max_consecutive_rises = max(max_consecutive_rises, consecutive_rises)
                                    else:
                                        consecutive_rises = 0
                                has_rising_trend = max_consecutive_rises >= 3
                            
                            is_rising_trend = False
                            slope = 0.0
                            if len(target_values) > 1:
                                target_indices_local = np.arange(len(target_values))
                                slope = np.polyfit(target_indices_local, target_values, 1)[0]
                                is_rising_trend = slope > 0
                            
                            has_rise_in_range = exceeds_peak or has_rising_trend or is_rising_trend
                            
                            if has_rise_in_range:
                                reasons = []
                                if exceeds_peak:
                                    reasons.append(f"有值超过峰值({max_in_range:.4f}>{center_value:.4f})")
                                if has_rising_trend:
                                    reasons.append(f"连续{max_consecutive_rises}个点上升")
                                if is_rising_trend:
                                    reasons.append(f"整体趋势上升(斜率={slope:.4f})")
                                excluded_peaks.append({
                                    "ratio": peak_ratio,
                                    "value": center_value,
                                    "reasons": reasons
                                })
                                continue
        
        # 调用find_all_peaks获取通过检查的峰值
        all_peaks = self.find_all_peaks(min_ratio=0.30, max_ratio=0.60, min_peak_height=0.3, window_size=5)
        
        # 输出排除的峰值
        if len(excluded_peaks) > 0:
            logger.info(f"   ❌ 排除了 {len(excluded_peaks)} 个峰值点（往后10%范围内有上升）:")
            for i, peak in enumerate(excluded_peaks, 1):
                reasons_str = ", ".join(peak["reasons"])
                logger.info(f"      {i}. {peak['ratio']*100:.2f}% (F1={peak['value']:.4f}) - 排除原因: {reasons_str}")
        
        if len(all_peaks) == 0:
            if len(excluded_peaks) > 0:
                logger.info(f"   ⚠️ 所有峰值点均被排除，未找到有效的断崖点")
                return {"method": "gradient", "detected": False, "reason": f"找到{len(excluded_peaks)}个峰值但均因往后10%范围内有上升而被排除"}
            logger.info("   ⚠️ 未找到峰值点")
            return {"method": "gradient", "detected": False, "reason": "未找到峰值点"}
        
        logger.info(f"   ✅ 找到 {len(all_peaks)} 个候选峰值点（通过往后10%上升检查）:")
        for i, peak in enumerate(all_peaks, 1):
            status = "✅ 持续下降" if peak["sustained_decline"] else "⚠️ 有反弹"
            logger.info(f"      {i}. {peak['ratio_percent']:.2f}% (F1={peak['value']:.4f}) - {status} (下降={peak['drop_percentage']:.1%})")
        
        # 优先选择"持续下降"的峰值
        sustained_peaks = [p for p in all_peaks if p["sustained_decline"]]
        
        if len(sustained_peaks) > 0:
            # 选择下降幅度最大的持续下降峰值
            best_peak = max(sustained_peaks, key=lambda x: x["drop_percentage"])
            cliff_ratio = best_peak["ratio"]
            logger.info(f"\n   ✅ 选择主要断崖点: {cliff_ratio:.2%} (持续下降，下降={best_peak['drop_percentage']:.1%})")
        else:
            # 如果没有持续下降的峰值，选择下降幅度最大的峰值
            best_peak = max(all_peaks, key=lambda x: x["drop_percentage"])
            cliff_ratio = best_peak["ratio"]
            logger.info(f"\n   ⚠️ 选择主要断崖点: {cliff_ratio:.2%} (有反弹，但下降幅度最大={best_peak['drop_percentage']:.1%})")
        
        return {
            "method": "gradient",
            "detected": True,
            "cliff_ratio": float(cliff_ratio),
            "cliff_ratio_percent": float(cliff_ratio * 100),
            "peak_value": float(best_peak["value"]),
            "drop_percentage": float(best_peak["drop_percentage"]),
            "index": int(best_peak["index"]),
            "sustained_decline": bool(best_peak["sustained_decline"]),
            "all_peaks": all_peaks  # 包含所有候选峰值点
        }
    
    def method2_second_derivative(self) -> Dict:
        """
        方法2: 二阶导数分析 - 找所有峰值点，选择持续下降的峰值作为断崖点
        
        策略：与方法1相同，找到所有局部峰值，选择"峰值后持续下降且不反弹"的峰值
        
        Returns:
            检测结果字典，包含所有候选峰值点
        """
        logger.info("\n【方法2: 二阶导数分析（多峰值检测）】")
        
        if len(self.sorted_values) < 20:
            return {"method": "second_derivative", "detected": False, "reason": "样本数太少"}
        
        # 找到所有峰值点（在30%-60%范围内）
        # 使用与方法1相同的详细日志逻辑
        logger.info("   正在搜索峰值点（30%-60%范围）...")
        
        region_mask = (self.sorted_ratios >= 0.30) & (self.sorted_ratios <= 0.60)
        if not np.any(region_mask):
            logger.info("   未找到30%-60%范围内的数据")
            return {"method": "second_derivative", "detected": False, "reason": "未找到峰值点"}
        
        region_indices = np.where(region_mask)[0]
        region_ratios = self.sorted_ratios[region_indices]
        region_values = self.sorted_values[region_indices]
        
        excluded_peaks = []
        window_size = 5
        min_peak_height = 0.3
        
        for i in range(window_size, len(region_values) - window_size):
            center_idx = region_indices[i]
            center_value = region_values[i]
            peak_ratio = region_ratios[i]
            
            left_window = region_values[i - window_size:i]
            right_window = region_values[i + 1:i + window_size + 1]
            
            if len(left_window) > 0 and len(right_window) > 0:
                if center_value >= np.max(left_window) and center_value >= np.max(right_window):
                    if center_value >= min_peak_height:
                        target_ratio = peak_ratio + 0.10
                        target_mask = (self.sorted_ratios > peak_ratio) & (self.sorted_ratios <= target_ratio)
                        target_indices = np.where(target_mask)[0]
                        
                        if len(target_indices) > 0:
                            target_values = self.sorted_values[target_indices]
                            max_in_range = np.max(target_values)
                            exceeds_peak = max_in_range > center_value
                            
                            has_rising_trend = False
                            max_consecutive_rises = 0
                            if len(target_values) > 2:
                                diffs = np.diff(target_values)
                                consecutive_rises = 0
                                for diff in diffs:
                                    if diff > 0:
                                        consecutive_rises += 1
                                        max_consecutive_rises = max(max_consecutive_rises, consecutive_rises)
                                    else:
                                        consecutive_rises = 0
                                has_rising_trend = max_consecutive_rises >= 3
                            
                            is_rising_trend = False
                            slope = 0.0
                            if len(target_values) > 1:
                                target_indices_local = np.arange(len(target_values))
                                slope = np.polyfit(target_indices_local, target_values, 1)[0]
                                is_rising_trend = slope > 0
                            
                            has_rise_in_range = exceeds_peak or has_rising_trend or is_rising_trend
                            
                            if has_rise_in_range:
                                reasons = []
                                if exceeds_peak:
                                    reasons.append(f"有值超过峰值({max_in_range:.4f}>{center_value:.4f})")
                                if has_rising_trend:
                                    reasons.append(f"连续{max_consecutive_rises}个点上升")
                                if is_rising_trend:
                                    reasons.append(f"整体趋势上升(斜率={slope:.4f})")
                                excluded_peaks.append({
                                    "ratio": peak_ratio,
                                    "value": center_value,
                                    "reasons": reasons
                                })
        
        all_peaks = self.find_all_peaks(min_ratio=0.30, max_ratio=0.60, min_peak_height=0.3, window_size=5)
        
        if len(excluded_peaks) > 0:
            logger.info(f"   ❌ 排除了 {len(excluded_peaks)} 个峰值点（往后10%范围内有上升）:")
            for i, peak in enumerate(excluded_peaks, 1):
                reasons_str = ", ".join(peak["reasons"])
                logger.info(f"      {i}. {peak['ratio']*100:.2f}% (F1={peak['value']:.4f}) - 排除原因: {reasons_str}")
        
        if len(all_peaks) == 0:
            if len(excluded_peaks) > 0:
                logger.info(f"   ⚠️ 所有峰值点均被排除，未找到有效的断崖点")
                return {"method": "second_derivative", "detected": False, "reason": f"找到{len(excluded_peaks)}个峰值但均因往后10%范围内有上升而被排除"}
            logger.info("   ⚠️ 未找到峰值点")
            return {"method": "second_derivative", "detected": False, "reason": "未找到峰值点"}
        
        logger.info(f"   ✅ 找到 {len(all_peaks)} 个候选峰值点（通过往后10%上升检查）:")
        for i, peak in enumerate(all_peaks, 1):
            status = "✅ 持续下降" if peak["sustained_decline"] else "⚠️ 有反弹"
            logger.info(f"      {i}. {peak['ratio_percent']:.2f}% (F1={peak['value']:.4f}) - {status} (下降={peak['drop_percentage']:.1%})")
        
        # 优先选择"持续下降"的峰值
        sustained_peaks = [p for p in all_peaks if p["sustained_decline"]]
        
        if len(sustained_peaks) > 0:
            # 选择下降幅度最大的持续下降峰值
            best_peak = max(sustained_peaks, key=lambda x: x["drop_percentage"])
            cliff_ratio = best_peak["ratio"]
            logger.info(f"\n   ✅ 选择主要断崖点: {cliff_ratio:.2%} (持续下降，下降={best_peak['drop_percentage']:.1%})")
        else:
            # 如果没有持续下降的峰值，选择下降幅度最大的峰值
            best_peak = max(all_peaks, key=lambda x: x["drop_percentage"])
            cliff_ratio = best_peak["ratio"]
            logger.info(f"\n   ⚠️ 选择主要断崖点: {cliff_ratio:.2%} (有反弹，但下降幅度最大={best_peak['drop_percentage']:.1%})")
        
        return {
            "method": "second_derivative",
            "detected": True,
            "cliff_ratio": float(cliff_ratio),
            "cliff_ratio_percent": float(cliff_ratio * 100),
            "peak_value": float(best_peak["value"]),
            "drop_percentage": float(best_peak["drop_percentage"]),
            "index": int(best_peak["index"]),
            "sustained_decline": bool(best_peak["sustained_decline"]),
            "all_peaks": all_peaks  # 包含所有候选峰值点
        }
    
    def method3_binned_statistics(self, n_bins: int = 20) -> Dict:
        """
        方法3: 分箱统计 - 找峰值后的性能下降最大的区间边界
        
        策略：先找到性能峰值所在的箱，然后在峰值箱之后寻找下降最大的边界
        
        Args:
            n_bins: 分箱数量
            
        Returns:
            检测结果字典
        """
        logger.info(f"\n【方法3: 分箱统计 (n_bins={n_bins})】")
        
        if len(self.sorted_values) < n_bins:
            return {"method": "binned", "detected": False, "reason": "样本数少于分箱数"}
        
        # 创建分箱
        bin_edges = np.linspace(self.sorted_ratios[0], self.sorted_ratios[-1], n_bins + 1)
        bin_indices = np.digitize(self.sorted_ratios, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        # 计算每个箱的平均性能
        bin_means = []
        bin_centers = []
        for i in range(n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                bin_means.append(np.mean(self.sorted_values[mask]))
                bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            else:
                bin_means.append(np.nan)
                bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
        
        bin_means = np.array(bin_means)
        bin_centers = np.array(bin_centers)
        
        # 找到性能峰值所在的箱（在35%-55%范围内）
        valid_mask = ~np.isnan(bin_means)
        if np.sum(valid_mask) < 3:
            return {"method": "binned", "detected": False, "reason": "有效分箱数太少"}
        
        valid_means = bin_means[valid_mask]
        valid_centers = bin_centers[valid_mask]
        valid_indices = np.where(valid_mask)[0]
        
        # 在35%-55%范围内找峰值
        critical_bin_mask = (valid_centers >= 0.35) & (valid_centers <= 0.55)
        if np.any(critical_bin_mask):
            critical_means = valid_means[critical_bin_mask]
            peak_bin_idx_in_critical = np.argmax(critical_means)
            peak_bin_idx = valid_indices[critical_bin_mask][peak_bin_idx_in_critical]
        else:
            peak_bin_idx = valid_indices[np.argmax(valid_means)]
        
        peak_ratio = bin_centers[peak_bin_idx]
        peak_value = bin_means[peak_bin_idx]
        logger.info(f"   检测到性能峰值箱: {peak_ratio:.2%}, F1={peak_value:.4f}")
        
        # 将峰值箱的中心作为断崖点
        cliff_ratio = peak_ratio
        
        # 计算性能变化（峰值箱 vs 下降后）
        before_perf = peak_value
        # 找到下降后的箱（看后续多个箱，判断是否持续下降）
        post_peak_bin_idx = peak_bin_idx + 1
        if post_peak_bin_idx < len(valid_means):
            # 看后续3个箱的平均值，判断是否持续下降
            post_peak_bins = valid_means[post_peak_bin_idx:min(post_peak_bin_idx + 3, len(valid_means))]
            after_perf = np.mean(post_peak_bins) if len(post_peak_bins) > 0 else valid_means[post_peak_bin_idx]
            max_rebound = np.max(post_peak_bins) if len(post_peak_bins) > 0 else after_perf
        else:
            after_perf = valid_means[-1]
            max_rebound = after_perf
        
        drop_percentage = (before_perf - after_perf) / before_perf if before_perf > 0 else 0
        
        # 判断是否持续下降：后续均值低于峰值，且最大反弹不超过峰值的95%
        sustained_decline = (after_perf < before_perf * 0.9) and (max_rebound < before_perf * 0.95)
        
        # 确保下降幅度足够大（>5%）
        if drop_percentage <= 0.05:
            return {"method": "binned", "detected": False, 
                   "reason": f"峰值后下降幅度太小（下降={drop_percentage:.1%}，需要>5%）"}
        
        status = "✅ 持续下降" if sustained_decline else "⚠️ 有反弹"
        logger.info(f"   检测到的临界比率: {cliff_ratio:.4f} ({cliff_ratio*100:.2f}%)")
        logger.info(f"   断崖前性能: {before_perf:.4f}")
        logger.info(f"   断崖后性能: {after_perf:.4f}")
        logger.info(f"   性能下降: {drop_percentage:.1%} ({status})")
        
        return {
            "method": "binned",
            "detected": True,
            "cliff_ratio": float(cliff_ratio),
            "cliff_ratio_percent": float(cliff_ratio * 100),
            "peak_value": float(peak_value),
            "before_performance": float(before_perf),
            "after_performance": float(after_perf),
            "drop_percentage": float(drop_percentage),
            "sustained_decline": bool(sustained_decline),
            "n_bins": n_bins
        }
    
    def method4_percentile_threshold(self, threshold_percentile: float = 0.1) -> Dict:
        """
        方法4: 百分位阈值 - 找所有峰值点，选择持续下降的峰值作为断崖点
        
        策略：与方法1相同，找到所有局部峰值，选择"峰值后持续下降且不反弹"的峰值
        
        Args:
            threshold_percentile: 性能阈值百分位（未使用，保持兼容性）
            
        Returns:
            检测结果字典，包含所有候选峰值点
        """
        logger.info(f"\n【方法4: 百分位阈值（多峰值检测）】")
        
        if len(self.sorted_values) < 10:
            return {"method": "percentile", "detected": False, "reason": "样本数太少"}
        
        # 找到所有峰值点（在30%-60%范围内）
        # 使用与方法1相同的详细日志逻辑
        logger.info("   正在搜索峰值点（30%-60%范围）...")
        
        region_mask = (self.sorted_ratios >= 0.30) & (self.sorted_ratios <= 0.60)
        if not np.any(region_mask):
            logger.info("   未找到30%-60%范围内的数据")
            return {"method": "percentile", "detected": False, "reason": "未找到峰值点"}
        
        region_indices = np.where(region_mask)[0]
        region_ratios = self.sorted_ratios[region_indices]
        region_values = self.sorted_values[region_indices]
        
        excluded_peaks = []
        window_size = 5
        min_peak_height = 0.3
        
        for i in range(window_size, len(region_values) - window_size):
            center_idx = region_indices[i]
            center_value = region_values[i]
            peak_ratio = region_ratios[i]
            
            left_window = region_values[i - window_size:i]
            right_window = region_values[i + 1:i + window_size + 1]
            
            if len(left_window) > 0 and len(right_window) > 0:
                if center_value >= np.max(left_window) and center_value >= np.max(right_window):
                    if center_value >= min_peak_height:
                        target_ratio = peak_ratio + 0.10
                        target_mask = (self.sorted_ratios > peak_ratio) & (self.sorted_ratios <= target_ratio)
                        target_indices = np.where(target_mask)[0]
                        
                        if len(target_indices) > 0:
                            target_values = self.sorted_values[target_indices]
                            max_in_range = np.max(target_values)
                            exceeds_peak = max_in_range > center_value
                            
                            has_rising_trend = False
                            max_consecutive_rises = 0
                            if len(target_values) > 2:
                                diffs = np.diff(target_values)
                                consecutive_rises = 0
                                for diff in diffs:
                                    if diff > 0:
                                        consecutive_rises += 1
                                        max_consecutive_rises = max(max_consecutive_rises, consecutive_rises)
                                    else:
                                        consecutive_rises = 0
                                has_rising_trend = max_consecutive_rises >= 3
                            
                            is_rising_trend = False
                            slope = 0.0
                            if len(target_values) > 1:
                                target_indices_local = np.arange(len(target_values))
                                slope = np.polyfit(target_indices_local, target_values, 1)[0]
                                is_rising_trend = slope > 0
                            
                            has_rise_in_range = exceeds_peak or has_rising_trend or is_rising_trend
                            
                            if has_rise_in_range:
                                reasons = []
                                if exceeds_peak:
                                    reasons.append(f"有值超过峰值({max_in_range:.4f}>{center_value:.4f})")
                                if has_rising_trend:
                                    reasons.append(f"连续{max_consecutive_rises}个点上升")
                                if is_rising_trend:
                                    reasons.append(f"整体趋势上升(斜率={slope:.4f})")
                                excluded_peaks.append({
                                    "ratio": peak_ratio,
                                    "value": center_value,
                                    "reasons": reasons
                                })
        
        all_peaks = self.find_all_peaks(min_ratio=0.30, max_ratio=0.60, min_peak_height=0.3, window_size=5)
        
        if len(excluded_peaks) > 0:
            logger.info(f"   ❌ 排除了 {len(excluded_peaks)} 个峰值点（往后10%范围内有上升）:")
            for i, peak in enumerate(excluded_peaks, 1):
                reasons_str = ", ".join(peak["reasons"])
                logger.info(f"      {i}. {peak['ratio']*100:.2f}% (F1={peak['value']:.4f}) - 排除原因: {reasons_str}")
        
        if len(all_peaks) == 0:
            if len(excluded_peaks) > 0:
                logger.info(f"   ⚠️ 所有峰值点均被排除，未找到有效的断崖点")
                return {"method": "percentile", "detected": False, "reason": f"找到{len(excluded_peaks)}个峰值但均因往后10%范围内有上升而被排除"}
            logger.info("   ⚠️ 未找到峰值点")
            return {"method": "percentile", "detected": False, "reason": "未找到峰值点"}
        
        logger.info(f"   找到 {len(all_peaks)} 个候选峰值点:")
        for i, peak in enumerate(all_peaks, 1):
            status = "✅ 持续下降" if peak["sustained_decline"] else "⚠️ 有反弹"
            logger.info(f"      {i}. {peak['ratio_percent']:.2f}% (F1={peak['value']:.4f}) - {status} (下降={peak['drop_percentage']:.1%})")
        
        # 优先选择"持续下降"的峰值
        sustained_peaks = [p for p in all_peaks if p["sustained_decline"]]
        
        if len(sustained_peaks) > 0:
            # 选择下降幅度最大的持续下降峰值
            best_peak = max(sustained_peaks, key=lambda x: x["drop_percentage"])
            cliff_ratio = best_peak["ratio"]
            logger.info(f"\n   ✅ 选择主要断崖点: {cliff_ratio:.2%} (持续下降，下降={best_peak['drop_percentage']:.1%})")
        else:
            # 如果没有持续下降的峰值，选择下降幅度最大的峰值
            best_peak = max(all_peaks, key=lambda x: x["drop_percentage"])
            cliff_ratio = best_peak["ratio"]
            logger.info(f"\n   ⚠️ 选择主要断崖点: {cliff_ratio:.2%} (有反弹，但下降幅度最大={best_peak['drop_percentage']:.1%})")
        
        return {
            "method": "percentile",
            "detected": True,
            "cliff_ratio": float(cliff_ratio),
            "cliff_ratio_percent": float(cliff_ratio * 100),
            "peak_value": float(best_peak["value"]),
            "drop_percentage": float(best_peak["drop_percentage"]),
            "index": int(best_peak["index"]),
            "sustained_decline": bool(best_peak["sustained_decline"]),
            "all_peaks": all_peaks  # 包含所有候选峰值点
        }
    
    def method5_sliding_window(self, window_size: int = 50, drop_threshold: float = 0.3) -> Dict:
        """
        方法5: 滑动窗口 - 找所有峰值点，选择持续下降的峰值作为断崖点
        
        策略：与方法1相同，找到所有局部峰值，选择"峰值后持续下降且不反弹"的峰值
        
        Args:
            window_size: 窗口大小（未使用，保持兼容性）
            drop_threshold: 下降阈值（未使用，保持兼容性）
            
        Returns:
            检测结果字典，包含所有候选峰值点
        """
        logger.info(f"\n【方法5: 滑动窗口（多峰值检测）】")
        
        if len(self.sorted_values) < 20:
            return {"method": "sliding_window", "detected": False, "reason": "样本数太少"}
        
        # 找到所有峰值点（在30%-60%范围内）
        # 使用与方法1相同的详细日志逻辑
        logger.info("   正在搜索峰值点（30%-60%范围）...")
        
        region_mask = (self.sorted_ratios >= 0.30) & (self.sorted_ratios <= 0.60)
        if not np.any(region_mask):
            logger.info("   未找到30%-60%范围内的数据")
            return {"method": "sliding_window", "detected": False, "reason": "未找到峰值点"}
        
        region_indices = np.where(region_mask)[0]
        region_ratios = self.sorted_ratios[region_indices]
        region_values = self.sorted_values[region_indices]
        
        excluded_peaks = []
        window_size = 5
        min_peak_height = 0.3
        
        for i in range(window_size, len(region_values) - window_size):
            center_idx = region_indices[i]
            center_value = region_values[i]
            peak_ratio = region_ratios[i]
            
            left_window = region_values[i - window_size:i]
            right_window = region_values[i + 1:i + window_size + 1]
            
            if len(left_window) > 0 and len(right_window) > 0:
                if center_value >= np.max(left_window) and center_value >= np.max(right_window):
                    if center_value >= min_peak_height:
                        target_ratio = peak_ratio + 0.10
                        target_mask = (self.sorted_ratios > peak_ratio) & (self.sorted_ratios <= target_ratio)
                        target_indices = np.where(target_mask)[0]
                        
                        if len(target_indices) > 0:
                            target_values = self.sorted_values[target_indices]
                            max_in_range = np.max(target_values)
                            exceeds_peak = max_in_range > center_value
                            
                            has_rising_trend = False
                            max_consecutive_rises = 0
                            if len(target_values) > 2:
                                diffs = np.diff(target_values)
                                consecutive_rises = 0
                                for diff in diffs:
                                    if diff > 0:
                                        consecutive_rises += 1
                                        max_consecutive_rises = max(max_consecutive_rises, consecutive_rises)
                                    else:
                                        consecutive_rises = 0
                                has_rising_trend = max_consecutive_rises >= 3
                            
                            is_rising_trend = False
                            slope = 0.0
                            if len(target_values) > 1:
                                target_indices_local = np.arange(len(target_values))
                                slope = np.polyfit(target_indices_local, target_values, 1)[0]
                                is_rising_trend = slope > 0
                            
                            has_rise_in_range = exceeds_peak or has_rising_trend or is_rising_trend
                            
                            if has_rise_in_range:
                                reasons = []
                                if exceeds_peak:
                                    reasons.append(f"有值超过峰值({max_in_range:.4f}>{center_value:.4f})")
                                if has_rising_trend:
                                    reasons.append(f"连续{max_consecutive_rises}个点上升")
                                if is_rising_trend:
                                    reasons.append(f"整体趋势上升(斜率={slope:.4f})")
                                excluded_peaks.append({
                                    "ratio": peak_ratio,
                                    "value": center_value,
                                    "reasons": reasons
                                })
        
        all_peaks = self.find_all_peaks(min_ratio=0.30, max_ratio=0.60, min_peak_height=0.3, window_size=5)
        
        if len(excluded_peaks) > 0:
            logger.info(f"   ❌ 排除了 {len(excluded_peaks)} 个峰值点（往后10%范围内有上升）:")
            for i, peak in enumerate(excluded_peaks, 1):
                reasons_str = ", ".join(peak["reasons"])
                logger.info(f"      {i}. {peak['ratio']*100:.2f}% (F1={peak['value']:.4f}) - 排除原因: {reasons_str}")
        
        if len(all_peaks) == 0:
            if len(excluded_peaks) > 0:
                logger.info(f"   ⚠️ 所有峰值点均被排除，未找到有效的断崖点")
                return {"method": "sliding_window", "detected": False, "reason": f"找到{len(excluded_peaks)}个峰值但均因往后10%范围内有上升而被排除"}
            logger.info("   ⚠️ 未找到峰值点")
            return {"method": "sliding_window", "detected": False, "reason": "未找到峰值点"}
        
        logger.info(f"   找到 {len(all_peaks)} 个候选峰值点:")
        for i, peak in enumerate(all_peaks, 1):
            status = "✅ 持续下降" if peak["sustained_decline"] else "⚠️ 有反弹"
            logger.info(f"      {i}. {peak['ratio_percent']:.2f}% (F1={peak['value']:.4f}) - {status} (下降={peak['drop_percentage']:.1%})")
        
        # 优先选择"持续下降"的峰值
        sustained_peaks = [p for p in all_peaks if p["sustained_decline"]]
        
        if len(sustained_peaks) > 0:
            # 选择下降幅度最大的持续下降峰值
            best_peak = max(sustained_peaks, key=lambda x: x["drop_percentage"])
            cliff_ratio = best_peak["ratio"]
            logger.info(f"\n   ✅ 选择主要断崖点: {cliff_ratio:.2%} (持续下降，下降={best_peak['drop_percentage']:.1%})")
        else:
            # 如果没有持续下降的峰值，选择下降幅度最大的峰值
            best_peak = max(all_peaks, key=lambda x: x["drop_percentage"])
            cliff_ratio = best_peak["ratio"]
            logger.info(f"\n   ⚠️ 选择主要断崖点: {cliff_ratio:.2%} (有反弹，但下降幅度最大={best_peak['drop_percentage']:.1%})")
        
        return {
            "method": "sliding_window",
            "detected": True,
            "cliff_ratio": float(cliff_ratio),
            "cliff_ratio_percent": float(cliff_ratio * 100),
            "peak_value": float(best_peak["value"]),
            "drop_percentage": float(best_peak["drop_percentage"]),
            "index": int(best_peak["index"]),
            "sustained_decline": bool(best_peak["sustained_decline"]),
            "all_peaks": all_peaks  # 包含所有候选峰值点
        }
    
    def detect_all_methods(self, metric_name: str = 'f1') -> Dict:
        """
        使用所有方法检测断崖点
        
        Args:
            metric_name: 指标名称
            
        Returns:
            所有方法的检测结果
        """
        if not self.load_data():
            return {}
        
        if not self.extract_metric_data(metric_name):
            return {}
        
        logger.info("\n" + "="*60)
        logger.info("开始使用多种方法检测断崖点...")
        logger.info("="*60)
        
        results = {}
        
        # 方法1: 梯度分析
        results['gradient'] = self.method1_gradient_analysis()
        
        # 方法2: 二阶导数
        results['second_derivative'] = self.method2_second_derivative()
        
        # 方法3: 分箱统计
        results['binned'] = self.method3_binned_statistics(n_bins=20)
        
        # 方法4: 百分位阈值
        results['percentile'] = self.method4_percentile_threshold(threshold_percentile=0.1)
        
        # 方法5: 滑动窗口
        results['sliding_window'] = self.method5_sliding_window(window_size=50, drop_threshold=0.3)
        
        # 综合结果
        logger.info("\n" + "="*60)
        logger.info("【综合结果】")
        logger.info("="*60)
        
        detected_results = {k: v for k, v in results.items() if v.get('detected', False)}
        
        if detected_results:
            # 收集所有候选峰值点（从所有方法中）
            all_candidate_peaks = []
            for method, result in detected_results.items():
                if 'all_peaks' in result:
                    for peak in result['all_peaks']:
                        # 避免重复（基于ratio）
                        if not any(abs(p['ratio'] - peak['ratio']) < 0.001 for p in all_candidate_peaks):
                            all_candidate_peaks.append(peak)
            
            # 按ratio排序
            all_candidate_peaks.sort(key=lambda x: x['ratio'])
            
            # 计算所有检测到的断崖点的平均值
            cliff_ratios = [v['cliff_ratio'] for v in detected_results.values()]
            avg_cliff_ratio = np.mean(cliff_ratios)
            std_cliff_ratio = np.std(cliff_ratios)
            
            logger.info(f"\n✅ 共 {len(detected_results)}/{len(results)} 种方法检测到断崖点")
            logger.info(f"\n各方法检测到的临界比率:")
            for method, result in detected_results.items():
                sustained = "✅持续下降" if result.get('sustained_decline', False) else "⚠️有反弹"
                logger.info(f"  - {method:20s}: {result['cliff_ratio_percent']:6.2f}% (下降 {result['drop_percentage']:.1%}, {sustained})")
            
            # 显示所有候选峰值点
            if len(all_candidate_peaks) > 0:
                logger.info(f"\n📋 所有候选峰值点 ({len(all_candidate_peaks)} 个):")
                for i, peak in enumerate(all_candidate_peaks, 1):
                    status = "✅ 持续下降" if peak['sustained_decline'] else "⚠️ 有反弹"
                    logger.info(f"  {i}. {peak['ratio_percent']:6.2f}% (F1={peak['value']:.4f}) - {status} (下降={peak['drop_percentage']:.1%})")
            
            # 计算最终推荐值（使用中位数，更稳健）
            median_cliff_ratio = np.median(cliff_ratios)
            final_cliff_ratio = median_cliff_ratio
            final_cliff_ratio_percent = final_cliff_ratio * 100
            
            # 计算置信区间（如果标准差较小，说明方法结果一致）
            if std_cliff_ratio < 0.05:  # 标准差小于5%
                consistency = "高"
                confidence = "高"
            elif std_cliff_ratio < 0.10:  # 标准差小于10%
                consistency = "中"
                confidence = "中"
            else:
                consistency = "低"
                confidence = "低"
            
            logger.info(f"\n📊 统计信息:")
            logger.info(f"   平均临界比率: {avg_cliff_ratio*100:.2f}%")
            logger.info(f"   标准差: {std_cliff_ratio*100:.2f}%")
            logger.info(f"   范围: {min(cliff_ratios)*100:.2f}% - {max(cliff_ratios)*100:.2f}%")
            logger.info(f"   一致性: {consistency} (标准差 {std_cliff_ratio*100:.2f}%)")
            
            logger.info(f"\n" + "="*60)
            logger.info(f"🎯 【最终断崖点位置】")
            logger.info(f"   上下文百分比: {final_cliff_ratio_percent:.2f}%")
            logger.info(f"   置信度: {confidence}")
            logger.info(f"   基于 {len(detected_results)} 种方法的交叉验证")
            logger.info("="*60)
            
            results['summary'] = {
                "n_methods_detected": len(detected_results),
                "n_methods_total": len(results),
                # 最终单一结果
                "final_cliff_ratio": float(final_cliff_ratio),
                "final_cliff_ratio_percent": float(final_cliff_ratio_percent),
                # 统计信息（供参考）
                "average_cliff_ratio": float(avg_cliff_ratio),
                "average_cliff_ratio_percent": float(avg_cliff_ratio * 100),
                "median_cliff_ratio": float(median_cliff_ratio),
                "median_cliff_ratio_percent": float(median_cliff_ratio * 100),
                "std_cliff_ratio": float(std_cliff_ratio),
                "min_cliff_ratio": float(min(cliff_ratios)),
                "max_cliff_ratio": float(max(cliff_ratios)),
                "consistency": consistency,
                "confidence": confidence,
                "detected_methods": list(detected_results.keys()),
                # 所有候选峰值点
                "all_candidate_peaks": all_candidate_peaks
            }
        else:
            logger.warning("⚠️  所有方法均未检测到明显的断崖点")
            logger.info("\n" + "="*60)
            logger.info("❌ 未能确定断崖点位置")
            logger.info("="*60)
            results['summary'] = {
                "n_methods_detected": 0,
                "n_methods_total": len(results),
                "final_cliff_ratio": None,
                "final_cliff_ratio_percent": None,
                "message": "未检测到明显的断崖点"
            }
        
        return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="精确检测LLM断崖式降智点",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 检测默认结果文件
  python detect_cliff_point.py --results-dir results/mixed
  
  # 指定具体的JSON文件
  python detect_cliff_point.py --file results/mixed/natural_length_qwen2.5-7b_mixed_reading_comprehension.json
  
  # 指定指标名称
  python detect_cliff_point.py --file results/mixed/xxx.json --metric f1
        """
    )
    
    parser.add_argument('--file', type=str,
                       help='结果JSON文件路径（如果指定，则忽略--results-dir和--model等参数）')
    parser.add_argument('--results-dir', type=str, default='results/mixed',
                       help='结果目录（默认: results/mixed）')
    parser.add_argument('--model', type=str, default='qwen2.5-7b',
                       help='模型名称（默认: qwen2.5-7b）')
    parser.add_argument('--dataset', type=str, default='mixed',
                       help='数据集名称（默认: mixed）')
    parser.add_argument('--task', type=str, default='reading_comprehension',
                       help='任务类型（默认: reading_comprehension）')
    parser.add_argument('--metric', type=str, default='f1',
                       help='指标名称（默认: f1）')
    
    args = parser.parse_args()
    
    # 确定结果文件路径
    if args.file:
        results_file = Path(args.file)
    else:
        filename = f"natural_length_{args.model}_{args.dataset}_{args.task}.json"
        results_file = Path(args.results_dir) / filename
        
        # 如果不存在，尝试.bak文件
        if not results_file.exists():
            bak_file = results_file.with_suffix('.json.bak')
            if bak_file.exists():
                results_file = bak_file
                logger.info(f"使用备份文件: {bak_file}")
    
    if not results_file.exists():
        logger.error(f"结果文件不存在: {results_file}")
        logger.info(f"请检查文件路径，或使用 --file 参数指定具体文件")
        return
    
    logger.info(f"📁 结果文件: {results_file}")
    
    # 创建检测器并执行检测
    detector = CliffPointDetector(results_file)
    results = detector.detect_all_methods(
        metric_name=args.metric
    )
    
    # 保存结果
    if results:
        output_file = results_file.parent / f"cliff_point_analysis_{args.metric}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"\n💾 详细分析结果已保存到: {output_file}")
        
        # 输出最终单一结果（便于脚本调用）
        summary = results.get('summary', {})
        final_ratio = summary.get('final_cliff_ratio_percent')
        if final_ratio is not None:
            logger.info(f"\n📌 最终断崖点: {final_ratio:.2f}%")
            # 也保存到单独的文本文件，便于其他脚本读取
            result_txt_file = results_file.parent / f"cliff_point_final_{args.metric}.txt"
            with open(result_txt_file, 'w', encoding='utf-8') as f:
                f.write(f"{final_ratio:.2f}\n")
            logger.info(f"📄 最终结果已保存到: {result_txt_file}")
    
    logger.info("\n" + "="*60)
    logger.info("✅ 检测完成！")
    logger.info("="*60)


if __name__ == '__main__':
    main()

