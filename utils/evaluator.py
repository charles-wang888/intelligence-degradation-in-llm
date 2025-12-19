"""
评估模块
实现不同任务类型的评估指标计算
"""

import re
import json
from typing import Dict, List, Tuple, Optional
import logging
from collections import Counter

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    logging.warning("rouge_score未安装，ROUGE指标将不可用")

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False
    logging.warning("nltk未安装，BLEU指标将不可用")

logger = logging.getLogger(__name__)


class Evaluator:
    """评估器基类"""
    
    def evaluate(self, prediction: str, reference: str, **kwargs) -> Dict:
        """评估预测结果"""
        raise NotImplementedError


class InformationRetrievalEvaluator(Evaluator):
    """信息检索任务评估器（优化版，更适合生成式模型）"""
    
    def __init__(self):
        # 常见停用词（英文和中文）
        self.stopwords = {
            'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'should', 'could', 'may', 'might', 'can', 'must',
            '的', '了', '和', '是', '在', '有', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'
        }
    
    def evaluate(self, prediction: str, reference: str, **kwargs) -> Dict:
        """
        评估信息检索任务（改进版）
        
        Args:
            prediction: 模型预测结果
            reference: 参考答案
            context: 上下文（可选）
            
        Returns:
            包含recall, precision, f1, mrr的字典
        """
        # 标准化输入
        if not isinstance(prediction, str):
            prediction = str(prediction) if prediction else ""
        if not isinstance(reference, str):
            reference = str(reference) if reference else ""
        
        pred_lower = prediction.strip().lower()
        ref_lower = reference.strip().lower()
        
        # 如果reference为空，特殊处理
        if not ref_lower:
            if not pred_lower:
                return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "mrr": 1.0}
            else:
                return {"precision": 0.0, "recall": 1.0, "f1": 0.0, "mrr": 0.0}
        
        # 方法1: 子串匹配（最宽松）
        exact_match = 1.0 if ref_lower in pred_lower or pred_lower == ref_lower else 0.0
        
        # 方法2: Token级别的匹配（类似阅读理解）
        token_precision, token_recall, token_f1 = self._calculate_token_metrics(pred_lower, ref_lower)
        
        # 方法3: 关键信息匹配（提取有意义的词）
        key_precision, key_recall, key_f1 = self._calculate_key_info_metrics(prediction, reference)
        
        # 方法4: MRR计算（改进版）
        mrr = self._calculate_mrr(pred_lower, ref_lower)
        
        # 综合评估：取多种方法的最大值，更宽松的评估
        precision = max(token_precision, key_precision)
        recall = max(token_recall, key_recall)
        f1 = max(token_f1, key_f1)
        
        # 如果完全匹配，给予最高分
        if exact_match == 1.0:
            precision = max(precision, 1.0)
            recall = max(recall, 1.0)
            f1 = max(f1, 1.0)
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mrr": mrr
        }
    
    def _calculate_token_metrics(self, pred_lower: str, ref_lower: str) -> Tuple[float, float, float]:
        """计算基于token的精确率、召回率和F1"""
        pred_tokens = set(pred_lower.split())
        ref_tokens = set(ref_lower.split())
        
        # 移除停用词
        pred_tokens = {t for t in pred_tokens if t not in self.stopwords and len(t) > 1}
        ref_tokens = {t for t in ref_tokens if t not in self.stopwords and len(t) > 1}
        
        if len(ref_tokens) == 0:
            if len(pred_tokens) == 0:
                return (1.0, 1.0, 1.0)
            else:
                return (0.0, 1.0, 0.0)
        
        if len(pred_tokens) == 0:
            return (0.0, 0.0, 0.0)
        
        intersection = len(pred_tokens & ref_tokens)
        precision = intersection / len(pred_tokens)
        recall = intersection / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return (precision, recall, f1)
    
    def _calculate_key_info_metrics(self, prediction: str, reference: str) -> Tuple[float, float, float]:
        """计算基于关键信息的精确率、召回率和F1"""
        pred_info = self._extract_key_info(prediction)
        ref_info = self._extract_key_info(reference)
        
        if len(ref_info) == 0:
            if len(pred_info) == 0:
                return (1.0, 1.0, 1.0)
            else:
                return (0.0, 1.0, 0.0)
        
        if len(pred_info) == 0:
            return (0.0, 0.0, 0.0)
        
        intersection = len(set(pred_info) & set(ref_info))
        precision = intersection / len(pred_info)
        recall = intersection / len(ref_info)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return (precision, recall, f1)
    
    def _extract_key_info(self, text: str) -> List[str]:
        """提取关键信息（数字、专有名词、有意义的词）"""
        # 确保 text 是字符串类型
        if not isinstance(text, str):
            if isinstance(text, list):
                text = " ".join(str(item) for item in text if item)
            else:
                text = str(text) if text else ""
        
        if not text:
            return []
        
        key_info = []
        
        # 提取数字
        numbers = re.findall(r'\d+', text)
        key_info.extend(numbers)
        
        # 提取大写字母开头的词（专有名词）
        proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', text)
        key_info.extend([w.lower() for w in proper_nouns])
        
        # 提取所有有意义的词（去除停用词和单字符）
        # 使用正则表达式提取单词（支持英文和中文）
        words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())  # 英文单词（至少2个字符）
        words.extend(re.findall(r'[\u4e00-\u9fa5]+', text))  # 中文字符
        
        # 过滤停用词和单字符
        meaningful_words = [w for w in words if w not in self.stopwords and len(w) > 1]
        key_info.extend(meaningful_words)
        
        # 去重并返回
        return list(set(key_info))
    
    def _calculate_mrr(self, pred_lower: str, ref_lower: str) -> float:
        """计算MRR（改进版，考虑部分匹配）"""
        # 完全匹配
        if ref_lower in pred_lower or pred_lower == ref_lower:
            return 1.0
        
        # 部分匹配：检查reference的主要部分是否在prediction中
        ref_tokens = [t for t in ref_lower.split() if t not in self.stopwords and len(t) > 1]
        if len(ref_tokens) == 0:
            return 0.0
        
        # 计算有多少reference的token出现在prediction中
        matched_tokens = sum(1 for token in ref_tokens if token in pred_lower)
        match_ratio = matched_tokens / len(ref_tokens)
        
        # 如果匹配率超过50%，给予部分分数
        if match_ratio >= 0.5:
            return match_ratio
        
        return 0.0


class ReadingComprehensionEvaluator(Evaluator):
    """阅读理解任务评估器"""
    
    def evaluate(self, prediction: str, reference: str, **kwargs) -> Dict:
        """
        评估阅读理解任务
        
        Args:
            prediction: 模型预测结果
            reference: 参考答案
            
        Returns:
            包含f1的字典（只保留F1分数，移除accuracy和exact_match）
        """
        # 标准化输入
        pred_lower = prediction.strip().lower()
        ref_lower = reference.strip().lower()
        
        # F1分数（改进的token匹配，考虑部分匹配）
        f1 = self._calculate_f1(prediction, reference)
        
        return {
            "f1": f1
        }
    
    def _calculate_f1(self, prediction: str, reference: str) -> float:
        """
        计算F1分数（改进版，更适合生成式模型）
        考虑部分匹配和字符级别的匹配
        """
        pred_lower = prediction.strip().lower()
        ref_lower = reference.strip().lower()
        
        if not ref_lower:
            return 1.0 if not pred_lower else 0.0
        
        # 方法1: 基于token的F1（原有方法）
        pred_tokens = set(pred_lower.split())
        ref_tokens = set(ref_lower.split())
        
        if len(ref_tokens) == 0:
            return 1.0 if len(pred_tokens) == 0 else 0.0
        
        token_intersection = len(pred_tokens & ref_tokens)
        
        if token_intersection == 0:
            # 如果token完全不匹配，尝试字符级别的匹配
            return self._calculate_char_f1(pred_lower, ref_lower)
        
        token_precision = token_intersection / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
        token_recall = token_intersection / len(ref_tokens)
        token_f1 = 2 * token_precision * token_recall / (token_precision + token_recall) if (token_precision + token_recall) > 0 else 0.0
        
        # 方法2: 字符级别的F1（作为补充）
        char_f1 = self._calculate_char_f1(pred_lower, ref_lower)
        
        # 取两种方法的较大值，更宽松的评估
        return max(token_f1, char_f1)
    
    def _calculate_char_f1(self, prediction: str, reference: str) -> float:
        """
        计算基于字符的F1分数
        用于处理token不匹配但内容相关的情况
        """
        if not reference:
            return 1.0 if not prediction else 0.0
        
        # 检查reference是否作为子串出现在prediction中
        if reference in prediction:
            # 如果完全包含，给予较高的F1分数
            ref_len = len(reference)
            pred_len = len(prediction)
            if pred_len > 0:
                # 基于长度的比例计算F1
                recall = 1.0  # reference完全匹配
                precision = ref_len / pred_len  # 预测中的相关部分比例
                return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # 计算字符级别的重叠
        pred_chars = set(prediction)
        ref_chars = set(reference)
        
        if len(ref_chars) == 0:
            return 1.0 if len(pred_chars) == 0 else 0.0
        
        char_intersection = len(pred_chars & ref_chars)
        
        if char_intersection == 0:
            return 0.0
        
        char_precision = char_intersection / len(pred_chars) if len(pred_chars) > 0 else 0.0
        char_recall = char_intersection / len(ref_chars)
        char_f1 = 2 * char_precision * char_recall / (char_precision + char_recall) if (char_precision + char_recall) > 0 else 0.0
        
        return char_f1


class LogicalReasoningEvaluator(Evaluator):
    """逻辑推理任务评估器（优化版，更严格的评估）"""
    
    def __init__(self):
        # 常见停用词
        self.stopwords = {
            'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'should', 'could', 'may', 'might', 'can', 'must',
            '的', '了', '和', '是', '在', '有', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'
        }
        # 推理步骤指示词（中英文）
        self.step_indicators = [
            "首先", "然后", "接着", "最后", "因此", "所以", "因为", "由于",
            "first", "then", "next", "finally", "therefore", "thus", "because", "since",
            "步骤", "step", "推理", "reasoning", "结论", "conclusion"
        ]
    
    def evaluate(self, prediction: str, reference: str, **kwargs) -> Dict:
        """
        评估逻辑推理任务（改进版）
        
        Args:
            prediction: 模型预测结果
            reference: 参考答案
            reasoning_steps: 推理步骤数（可选）
            
        Returns:
            包含accuracy和reasoning_steps_correct的字典
        """
        # 标准化输入
        if not isinstance(prediction, str):
            prediction = str(prediction) if prediction else ""
        if not isinstance(reference, str):
            reference = str(reference) if reference else ""
        
        pred_lower = prediction.strip().lower()
        ref_lower = reference.strip().lower()
        
        # 准确率（改进版，使用多层次匹配）
        accuracy = self._calculate_accuracy(pred_lower, ref_lower, prediction, reference)
        
        # 推理步骤正确性（改进版）
        reasoning_steps_correct = self._check_reasoning_steps(prediction, kwargs.get("reasoning_steps", None))
        
        return {
            "accuracy": accuracy,
            "reasoning_steps_correct": reasoning_steps_correct
        }
    
    def _calculate_accuracy(self, pred_lower: str, ref_lower: str, prediction: str, reference: str) -> float:
        """
        计算准确率（改进版，使用多层次匹配策略）
        """
        if not ref_lower:
            return 1.0 if not pred_lower else 0.0
        
        # 方法1: 完全匹配或子串匹配（最宽松）
        if ref_lower in pred_lower or pred_lower == ref_lower:
            return 1.0
        
        # 方法2: Token级别的匹配（去除停用词）
        ref_tokens = {t for t in ref_lower.split() if t not in self.stopwords and len(t) > 1}
        pred_tokens = {t for t in pred_lower.split() if t not in self.stopwords and len(t) > 1}
        
        if len(ref_tokens) == 0:
            # 如果reference只有停用词，使用字符级别匹配
            return 1.0 if ref_lower in pred_lower else 0.0
        
        if len(pred_tokens) == 0:
            return 0.0
        
        # 计算token重叠率
        overlap = len(ref_tokens & pred_tokens)
        recall = overlap / len(ref_tokens)  # 召回率：有多少reference的token被匹配到
        
        # 方法3: 关键信息匹配（提取专有名词、数字等）
        ref_key_info = self._extract_key_info(reference)
        pred_key_info = self._extract_key_info(prediction)
        
        key_recall = 0.0
        if len(ref_key_info) > 0:
            key_overlap = len(set(ref_key_info) & set(pred_key_info))
            key_recall = key_overlap / len(ref_key_info)
        
        # 综合评估：取两种方法的较大值，但要求至少达到一定阈值
        final_recall = max(recall, key_recall)
        
        # 对于逻辑推理任务，要求较高的匹配度
        # 如果recall >= 0.8，认为基本正确
        # 如果recall >= 0.5，给予部分分数
        if final_recall >= 0.8:
            return 1.0
        elif final_recall >= 0.5:
            return 0.5 + (final_recall - 0.5) * 1.0  # 线性映射到[0.5, 1.0]
        else:
            return final_recall * 1.0  # 线性映射到[0, 0.5]
    
    def _extract_key_info(self, text: str) -> List[str]:
        """提取关键信息（数字、专有名词等）"""
        if not isinstance(text, str) or not text:
            return []
        
        key_info = []
        
        # 提取数字
        numbers = re.findall(r'\d+', text)
        key_info.extend(numbers)
        
        # 提取大写字母开头的词（专有名词）
        proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', text)
        key_info.extend([w.lower() for w in proper_nouns])
        
        # 提取所有有意义的词（去除停用词）
        words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
        words.extend(re.findall(r'[\u4e00-\u9fa5]+', text))
        
        meaningful_words = [w for w in words if w not in self.stopwords and len(w) > 1]
        key_info.extend(meaningful_words)
        
        return list(set(key_info))
    
    def _check_reasoning_steps(self, prediction: str, expected_steps: Optional[int]) -> float:
        """
        检查推理步骤（改进版）
        
        Args:
            prediction: 模型预测结果
            expected_steps: 期望的推理步骤数（None表示不检查具体数量）
            
        Returns:
            推理步骤正确性分数 [0.0, 1.0]
        """
        if not prediction:
            return 0.0
        
        # 统计预测中的推理步骤指示词
        found_indicators = [ind for ind in self.step_indicators if ind in prediction]
        found_count = len(found_indicators)
        
        # 如果找到了推理步骤指示词，说明有推理过程
        if found_count == 0:
            # 没有找到推理步骤指示词，检查是否有其他推理迹象
            # 检查是否有数字编号的步骤（如"1.", "2.", "步骤1"等）
            numbered_steps = len(re.findall(r'\d+[\.、]', prediction))
            if numbered_steps > 0:
                found_count = numbered_steps
        
        # 如果expected_steps为None，只检查是否有推理过程
        if expected_steps is None:
            # 至少需要找到1个推理步骤指示词或编号步骤
            if found_count > 0:
                # 根据找到的步骤数量给予分数（至少2个步骤才给满分）
                if found_count >= 2:
                    return 1.0
                else:
                    return 0.5  # 只有1个步骤，给予部分分数
            else:
                return 0.0  # 没有推理过程
        
        # 如果提供了expected_steps，检查步骤数量是否匹配
        if expected_steps == 0:
            # 如果期望0步，但找到了步骤，给予低分
            return 0.0 if found_count > 0 else 1.0
        
        # 计算步骤数量的匹配度
        if found_count == 0:
            return 0.0
        
        # 如果找到的步骤数接近期望值（±1），给予高分
        if abs(found_count - expected_steps) <= 1:
            return 1.0
        elif abs(found_count - expected_steps) <= 2:
            return 0.7  # 接近但不完全匹配
        else:
            return 0.3  # 步骤数量差异较大


class MathCalculationEvaluator(Evaluator):
    """数学计算任务评估器"""
    
    def evaluate(self, prediction: str, reference: str, **kwargs) -> Dict:
        """
        评估数学计算任务
        
        Args:
            prediction: 模型预测结果
            reference: 参考答案（数字）
            
        Returns:
            包含accuracy和calculation_correct的字典
        """
        # 提取预测中的数字
        pred_numbers = re.findall(r'\d+', prediction)
        ref_number = int(reference) if reference.isdigit() else 0
        
        # 计算准确率
        if pred_numbers:
            pred_number = int(pred_numbers[-1])  # 取最后一个数字
            accuracy = 1.0 if pred_number == ref_number else 0.0
        else:
            accuracy = 0.0
        
        # 计算正确性（是否包含正确答案）
        calculation_correct = 1.0 if str(ref_number) in prediction else 0.0
        
        return {
            "accuracy": accuracy,
            "calculation_correct": calculation_correct
        }


class SummarizationEvaluator(Evaluator):
    """
    摘要生成任务评估器（基于关键信息的加权评估）
    
    设计理念：
    1. 摘要的核心是包含关键信息（实体、数字、专有名词、重要概念）
    2. 关键信息比普通词更重要，应该给予更高权重
    3. 即使表达方式不同，只要关键信息覆盖完整，应该给予高分
    4. 结合token级别和关键信息级别的多层级评估
    """
    
    def __init__(self):
        # 常见停用词（英文和中文）
        self.stopwords = {
            'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'should', 'could', 'may', 'might', 'can', 'must',
            '的', '了', '和', '是', '在', '有', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'
        }
    
    def evaluate(self, prediction: str, reference: str, **kwargs) -> Dict:
        """
        评估摘要生成任务
        
        Args:
            prediction: 模型生成的摘要
            reference: 参考摘要（对于NarrativeQA/SQuAD，可能是短答案，不使用）
            context: 原始文本（重要！用于评估摘要是否覆盖了原文的关键信息）
            
        Returns:
            包含summary_score的字典（综合摘要质量分数）
        """
        # 标准化输入
        if not isinstance(prediction, str):
            prediction = str(prediction) if prediction else ""
        if not isinstance(reference, str):
            reference = str(reference) if reference else ""
        
        pred_lower = prediction.strip().lower()
        
        # 对于summarization任务，如果没有context，使用reference作为fallback
        context = kwargs.get("context", "")
        if not context:
            context = reference
        
        if not context:
            return {"summary_score": 1.0 if not pred_lower else 0.0}
        
        # 使用context作为参考，评估摘要是否覆盖了原文的关键信息
        # 这样就不需要真正的参考摘要了
        summary_score = self._calculate_summary_score(prediction, context)
        
        return {
            "summary_score": summary_score
        }
    
    def _calculate_summary_score(self, prediction: str, context: str) -> float:
        """
        计算综合摘要质量分数
        
        注意：这里context是原文，不是参考摘要
        评估摘要是否覆盖了原文的关键信息
        
        评估维度：
        1. 关键信息覆盖率（权重：0.7）- 最重要（摘要是否包含原文的关键信息）
        2. Token级别匹配（权重：0.2）
        3. 字符级别匹配（权重：0.1）
        
        Returns:
            综合分数 [0.0, 1.0]
        """
        pred_lower = prediction.strip().lower()
        context_lower = context.strip().lower()
        
        if not context_lower:
            return 1.0 if not pred_lower else 0.0
        
        # 维度1: 关键信息覆盖率（最重要，权重0.7）
        # 评估摘要是否覆盖了原文的关键信息
        key_info_score = self._calculate_key_info_coverage(prediction, context)
        
        # 维度2: Token级别匹配（权重0.2）
        token_score = self._calculate_token_score(pred_lower, context_lower)
        
        # 维度3: 字符级别匹配（权重0.1）
        char_score = self._calculate_char_score(pred_lower, context_lower)
        
        # 加权平均（更重视关键信息覆盖）
        summary_score = (
            key_info_score * 0.7 +
            token_score * 0.2 +
            char_score * 0.1
        )
        
        # 摘要应该比原文短，如果摘要长度合理，给予额外奖励
        pred_len = len(prediction)
        context_len = len(context)
        if context_len > 0:
            compression_ratio = pred_len / context_len
            # 理想的压缩比应该在0.1-0.3之间（摘要是原文的10%-30%）
            if 0.05 <= compression_ratio <= 0.5:
                summary_score = min(1.0, summary_score + 0.1)  # 给予额外奖励
            elif compression_ratio > 0.8:
                # 摘要太长（接近原文），可能不是好的摘要
                summary_score = summary_score * 0.8  # 降低分数
        
        return min(1.0, max(0.0, summary_score))
    
    def _calculate_key_info_coverage(self, prediction: str, context: str) -> float:
        """
        计算关键信息覆盖率（针对摘要任务优化）
        
        评估摘要是否覆盖了原文（context）的关键信息
        
        关键信息包括：
        - 数字（年份、数量、百分比等）
        - 专有名词（人名、地名、机构名等）
        - 重要概念（非停用词的有意义词）
        
        Returns:
            关键信息覆盖率 [0.0, 1.0]
        """
        # 提取关键信息
        context_key_info = self._extract_key_info(context)
        pred_key_info = self._extract_key_info(prediction)
        
        if len(context_key_info) == 0:
            # 如果context没有关键信息，回退到token匹配
            return self._calculate_token_score(prediction.lower(), context.lower())
        
        # 计算关键信息重叠
        context_key_set = set(context_key_info)
        pred_key_set = set(pred_key_info)
        
        # 计算覆盖率（recall）：摘要覆盖了多少原文的关键信息
        if len(context_key_set) == 0:
            return 1.0 if len(pred_key_set) == 0 else 0.0
        
        overlap = len(context_key_set & pred_key_set)
        recall = overlap / len(context_key_set)  # 这是最重要的：摘要是否覆盖了原文的关键信息
        
        # 计算精确率：摘要中的关键信息有多少来自原文
        precision = overlap / len(pred_key_set) if len(pred_key_set) > 0 else 0.0
        
        # F1分数
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # 对于摘要任务，recall更重要（是否覆盖了原文的关键信息）
        # 使用加权：recall权重0.9，precision权重0.1（更重视recall）
        key_info_score = recall * 0.9 + precision * 0.1
        
        # 如果F1很高，给予额外奖励
        if f1 > 0.5:  # 降低阈值
            key_info_score = min(1.0, key_info_score + 0.2)  # 增加奖励
        
        # 如果覆盖率超过30%，给予基础奖励
        if recall > 0.3:
            key_info_score = min(1.0, key_info_score + 0.1)
        
        # 确保返回值在[0,1]区间内
        return min(1.0, max(0.0, key_info_score))
    
    def _extract_key_info(self, text: str) -> List[str]:
        """
        提取关键信息
        
        包括：
        1. 数字（年份、数量、百分比等）
        2. 专有名词（大写字母开头的词）
        3. 重要概念（非停用词的有意义词，长度>=3）
        """
        if not isinstance(text, str) or not text:
            return []
        
        key_info = []
        
        # 1. 提取数字（包括年份、数量等）
        numbers = re.findall(r'\d+', text)
        key_info.extend(numbers)
        
        # 2. 提取专有名词（大写字母开头的词，至少2个字符）
        proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', text)
        key_info.extend([w.lower() for w in proper_nouns])
        
        # 3. 提取重要概念（非停用词的有意义词）
        # 英文单词（降低长度要求：从3个字符改为2个字符，更宽松）
        words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
        # 中文字词（至少2个字符）
        words.extend(re.findall(r'[\u4e00-\u9fa5]{2,}', text))
        
        # 过滤停用词（但保留更多词）
        meaningful_words = [
            w for w in words 
            if w not in self.stopwords and len(w) >= 2
        ]
        key_info.extend(meaningful_words)
        
        # 去重并返回
        return list(set(key_info))
    
    def _calculate_token_score(self, pred_lower: str, ref_lower: str) -> float:
        """
        计算基于token的匹配分数
        
        Returns:
            Token匹配分数 [0.0, 1.0]
        """
        if not ref_lower:
            return 1.0 if not pred_lower else 0.0
        
        pred_tokens = set(pred_lower.split())
        ref_tokens = set(ref_lower.split())
        
        # 移除停用词
        pred_tokens = {t for t in pred_tokens if t not in self.stopwords and len(t) > 1}
        ref_tokens = {t for t in ref_tokens if t not in self.stopwords and len(t) > 1}
        
        if len(ref_tokens) == 0:
            return 1.0 if len(pred_tokens) == 0 else 0.0
        
        if len(pred_tokens) == 0:
            return 0.0
        
        # 计算token重叠
        intersection = len(pred_tokens & ref_tokens)
        precision = intersection / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
        recall = intersection / len(ref_tokens)
        
        # F1分数
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # 确保返回值在[0,1]区间内
        return min(1.0, max(0.0, f1))
    
    def _calculate_char_score(self, pred_lower: str, ref_lower: str) -> float:
        """
        计算基于字符的匹配分数（作为补充）
        
        Returns:
            字符匹配分数 [0.0, 1.0]
        """
        if not ref_lower:
            return 1.0 if not pred_lower else 0.0
        
        # 检查reference是否作为子串出现在prediction中
        if ref_lower in pred_lower:
            # 如果完全包含，给予较高分数
            ref_len = len(ref_lower)
            pred_len = len(pred_lower)
            if pred_len > 0:
                recall = 1.0  # reference完全匹配
                precision = ref_len / pred_len
                return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # 计算字符级别的重叠
        pred_chars = set(pred_lower)
        ref_chars = set(ref_lower)
        
        if len(ref_chars) == 0:
            return 1.0 if len(pred_chars) == 0 else 0.0
        
        char_intersection = len(pred_chars & ref_chars)
        
        if char_intersection == 0:
            return 0.0
        
        char_precision = char_intersection / len(pred_chars) if len(pred_chars) > 0 else 0.0
        char_recall = char_intersection / len(ref_chars)
        char_f1 = 2 * char_precision * char_recall / (char_precision + char_recall) if (char_precision + char_recall) > 0 else 0.0
        
        return char_f1
    


def get_evaluator(task_type: str) -> Evaluator:
    """
    获取对应任务类型的评估器
    
    Args:
        task_type: 任务类型
        
    Returns:
        评估器实例
    """
    evaluators = {
        "information_retrieval": InformationRetrievalEvaluator(),
        "reading_comprehension": ReadingComprehensionEvaluator(),
        "logical_reasoning": LogicalReasoningEvaluator(),
        "math_calculation": MathCalculationEvaluator(),
        # "summarization": SummarizationEvaluator(),  # 已移除：NarrativeQA/SQuAD数据集不适合摘要任务
    }
    
    if task_type not in evaluators:
        raise ValueError(f"未知的任务类型: {task_type}")
    
    return evaluators[task_type]


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 测试阅读理解评估器
    evaluator = ReadingComprehensionEvaluator()
    result = evaluator.evaluate(
        prediction="人工智能是计算机科学的一个分支",
        reference="人工智能是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的系统。"
    )
    print("阅读理解评估结果:", result)

