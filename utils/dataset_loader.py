"""
数据集加载模块
支持多种开源数据集的加载和预处理
使用真实数据集，支持采样
"""

import json
import os
import requests
import random
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    logger.warning("datasets库未安装，将无法自动下载SQuAD数据集。请运行: pip install datasets")


class DatasetLoader:
    """数据集加载器基类"""
    
    def __init__(self, data_dir: str = "data", max_samples: Optional[int] = None, random_seed: int = 42):
        """
        初始化数据集加载器
        
        Args:
            data_dir: 数据存储目录
            max_samples: 最大采样数量（None表示使用全部数据）
            random_seed: 随机种子，用于可复现的采样
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.max_samples = max_samples
        self.random_seed = random_seed
    
    def load(self) -> List[Dict]:
        """加载数据集"""
        raise NotImplementedError
    
    def _sample_data(self, data: List[Dict]) -> List[Dict]:
        """
        从数据集中采样指定数量的样本
        
        Args:
            data: 原始数据集
            
        Returns:
            采样后的数据集
        """
        if self.max_samples is None or len(data) <= self.max_samples:
            return data
        
        # 使用固定随机种子确保可复现
        random.seed(self.random_seed)
        sampled = random.sample(data, self.max_samples)
        random.seed()  # 重置随机种子
        
        logger.info(f"从 {len(data)} 条数据中采样了 {len(sampled)} 条")
        return sampled
    
    def prepare_context(self, text: str, target_length: int) -> str:
        """
        准备指定长度的上下文
        
        Args:
            text: 原始文本
            target_length: 目标长度（token数，近似用字符数/4估算）
            
        Returns:
            截取或填充后的文本
        """
        # 简单估算：1 token ≈ 4 字符（中文）或 1 token ≈ 0.75 单词（英文）
        # 这里使用字符数/4作为近似
        approx_chars = target_length * 4
        
        if len(text) <= approx_chars:
            return text
        else:
            return text[:approx_chars]


class SQuADLoader(DatasetLoader):
    """SQuAD数据集加载器（使用真实数据集）"""
    
    def __init__(self, data_dir: str = "data", max_samples: Optional[int] = None, random_seed: int = 42):
        super().__init__(data_dir, max_samples, random_seed)
        self.dataset_name = "squad"
    
    def load(self) -> List[Dict]:
        """
        加载SQuAD数据集（真实数据集）
        
        Returns:
            数据集列表
        """
        dataset_path = self.data_dir / "squad.json"
        
        # 如果文件不存在，尝试从Hugging Face下载
        if not dataset_path.exists():
            logger.info(f"数据集文件不存在，尝试从Hugging Face下载...")
            if self._download_from_huggingface(dataset_path):
                logger.info("数据集下载成功")
            else:
                raise FileNotFoundError(
                    f"无法加载SQuAD数据集。\n"
                    f"请手动下载数据集并保存到: {dataset_path}\n"
                    f"或安装datasets库: pip install datasets\n"
                    f"然后运行代码会自动下载"
                )
        
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 标准化数据格式
            data = self._normalize_data(data)
            
            # 采样
            data = self._sample_data(data)
            
            logger.info(f"成功加载 {len(data)} 条数据")
            return data
        except Exception as e:
            logger.error(f"加载数据集失败: {e}")
            raise
    
    def _download_from_huggingface(self, save_path: Path) -> bool:
        """
        从Hugging Face下载SQuAD数据集
        
        Args:
            save_path: 保存路径
            
        Returns:
            是否下载成功
        """
        if not HAS_DATASETS:
            logger.error("需要安装datasets库: pip install datasets")
            return False
        
        try:
            logger.info("正在从Hugging Face下载SQuAD数据集（这可能需要一些时间）...")
            # 下载SQuAD v2.0数据集
            dataset = load_dataset("squad", "plain_text", split="train")
            
            # 转换为标准格式
            data = []
            for item in dataset:
                data.append({
                    "context": item["context"],
                    "question": item["question"],
                    "answers": item["answers"]["text"],
                    "id": item["id"]
                })
            
            # 保存到本地
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"下载完成，共 {len(data)} 条数据")
            return True
        except Exception as e:
            logger.error(f"从Hugging Face下载数据集失败: {e}")
            return False
    
    def _normalize_data(self, data: List[Dict]) -> List[Dict]:
        """
        标准化数据格式
        
        Args:
            data: 原始数据
            
        Returns:
            标准化后的数据
        """
        normalized = []
        
        for item in data:
            if isinstance(item, dict):
                # 确保有必要的字段
                context = item.get("context", "")
                question = item.get("question", "")
                answers = item.get("answers", [])
                
                if isinstance(answers, str):
                    answers = [answers]
                
                if context and question:
                    normalized.append({
                        "context": context,
                        "question": question,
                        "answers": answers if answers else [""],
                        "id": item.get("id", f"squad_{len(normalized)}")
                    })
        
        return normalized


class NarrativeQALoader(DatasetLoader):
    """NarrativeQA数据集加载器（长上下文阅读理解数据集）"""
    
    def __init__(self, data_dir: str = "data", max_samples: Optional[int] = None, random_seed: int = 42,
                 filter_answers: bool = False, min_answer_position: float = 0.10, max_answer_position: float = 0.90):
        super().__init__(data_dir, max_samples, random_seed)
        self.dataset_name = "narrativeqa"
        self.filter_answers = filter_answers
        self.min_answer_position = min_answer_position
        self.max_answer_position = max_answer_position
    
    def load(self) -> List[Dict]:
        """
        加载NarrativeQA数据集（长上下文数据集）
        
        NarrativeQA特点：
        - 平均上下文长度：6000-15000 tokens
        - 任务类型：基于完整书籍或电影剧本的问答
        - 适合观察长上下文下的性能变化
        
        Returns:
            数据集列表
        """
        dataset_path = self.data_dir / "narrativeqa.json"
        
        # 如果文件不存在，尝试从Hugging Face下载
        if not dataset_path.exists():
            logger.info(f"数据集文件不存在，尝试从Hugging Face下载...")
            if self._download_from_huggingface(dataset_path):
                logger.info("数据集下载成功")
            else:
                raise FileNotFoundError(
                    f"无法加载NarrativeQA数据集。\n"
                    f"请手动下载数据集并保存到: {dataset_path}\n"
                    f"或安装datasets库: pip install datasets\n"
                    f"然后运行代码会自动下载"
                )
        
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 标准化数据格式
            data = self._normalize_data(data)
            
            # 过滤掉上下文太短的样本（小于1000字符的）
            data = [item for item in data if len(item.get("context", "")) > 1000]
            logger.info(f"过滤后剩余 {len(data)} 条长上下文数据")
            
            # 答案筛选（如果启用）
            if self.filter_answers:
                logger.info(f"答案筛选已启用，将移除无答案或答案位置不合适的样本")
                data = self._filter_samples_with_answers(data)
            
            # 采样
            data = self._sample_data(data)
            
            logger.info(f"成功加载 {len(data)} 条数据")
            if data:
                avg_len = sum(len(item.get("context", "")) for item in data) / len(data)
                logger.info(f"平均上下文长度: {avg_len:.0f} 字符 (约 {avg_len/4:.0f} tokens)")
            
            return data
        except Exception as e:
            logger.error(f"加载数据集失败: {e}")
            raise
    
    def _download_from_huggingface(self, save_path: Path) -> bool:
        """
        从Hugging Face下载NarrativeQA数据集
        
        Args:
            save_path: 保存路径
            
        Returns:
            是否下载成功
        """
        if not HAS_DATASETS:
            logger.error("需要安装datasets库: pip install datasets")
            return False
        
        try:
            logger.info("正在从Hugging Face下载NarrativeQA数据集（这可能需要一些时间，数据集较大）...")
            # 下载NarrativeQA数据集（使用train split）
            dataset = load_dataset("narrativeqa", split="train")
            
            # 转换为标准格式
            data = []
            for item in dataset:
                # NarrativeQA的数据结构
                context = item.get("document", {}).get("text", "")
                question = item.get("question", {}).get("text", "")
                answers = item.get("answers", [])
                
                # 提取答案文本
                answer_texts = []
                if isinstance(answers, list):
                    for ans in answers:
                        if isinstance(ans, dict):
                            answer_texts.append(ans.get("text", ""))
                        elif isinstance(ans, str):
                            answer_texts.append(ans)
                elif isinstance(answers, str):
                    answer_texts = [answers]
                
                if context and question and answer_texts:
                    data.append({
                        "context": context,
                        "question": question,
                        "answers": answer_texts,
                        "id": item.get("id", f"narrativeqa_{len(data)}")
                    })
            
            # 保存到本地
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"下载完成，共 {len(data)} 条数据")
            return True
        except Exception as e:
            logger.error(f"从Hugging Face下载数据集失败: {e}")
            logger.error("如果下载失败，可以尝试手动下载：")
            logger.error("1. 访问 https://huggingface.co/datasets/narrativeqa")
            logger.error("2. 或使用命令: python -c \"from datasets import load_dataset; load_dataset('narrativeqa', split='train').to_json('data/narrativeqa.json')\"")
            return False
    
    def _normalize_data(self, data: List[Dict]) -> List[Dict]:
        """
        标准化数据格式
        
        Args:
            data: 原始数据
            
        Returns:
            标准化后的数据
        """
        normalized = []
        
        for item in data:
            if isinstance(item, dict):
                # 确保有必要的字段
                context = item.get("context", "")
                question = item.get("question", "")
                answers = item.get("answers", [])
                
                if isinstance(answers, str):
                    answers = [answers]
                
                if context and question and answers:
                    normalized.append({
                        "context": context,
                        "question": question,
                        "answers": answers if answers else [""],
                        "id": item.get("id", f"narrativeqa_{len(normalized)}")
                    })
        
        return normalized
    
    def _find_answer_position(self, context: str, answer: str) -> float:
        """
        查找答案在文本中的相对位置
        
        Args:
            context: 文本内容
            answer: 答案
            
        Returns:
            answer_ratio: 答案位置比例（0-1），如果找不到返回-1
        """
        if not context or not answer:
            return -1
        
        context_lower = context.lower()
        answer_lower = answer.lower()
        
        # 尝试精确匹配
        pos = context_lower.find(answer_lower)
        if pos != -1:
            return pos / len(context)
        
        # 尝试部分匹配（答案可能有多种形式）
        answer_words = answer_lower.split()
        if len(answer_words) > 0:
            # 查找第一个关键词（长度>3的词）
            for word in answer_words:
                if len(word) > 3:
                    pos = context_lower.find(word)
                    if pos != -1:
                        return pos / len(context)
        
        return -1  # 找不到
    
    def _filter_samples_with_answers(self, data: List[Dict]) -> List[Dict]:
        """
        筛选答案在文本中的样本
        
        Args:
            data: 原始数据
            
        Returns:
            筛选后的数据
        """
        logger.info("=" * 60)
        logger.info("开始答案筛选（移除无答案样本）")
        logger.info("=" * 60)
        
        filtered_data = []
        stats = {
            'total': len(data),
            'has_answer': 0,
            'position_ok': 0,
            'too_early': 0,
            'too_late': 0,
            'not_found': 0
        }
        
        for item in data:
            # 提取context
            context = item.get('document', {}).get('text', '') if isinstance(item.get('document'), dict) else item.get('document', '')
            if not context:
                context = item.get('context', '')
            
            # 提取answer
            answers = item.get('answers', [])
            if isinstance(answers, list) and len(answers) > 0:
                answer = answers[0]
                if isinstance(answer, dict):
                    answer = answer.get('text', '')
                answer = str(answer)
            else:
                answer = item.get('answer', '')
            
            if not answer:
                stats['not_found'] += 1
                continue
            
            # 查找答案位置
            answer_pos = self._find_answer_position(context, answer)
            
            if answer_pos < 0:
                stats['not_found'] += 1
            else:
                stats['has_answer'] += 1
                
                # 检查位置是否合适
                if answer_pos < self.min_answer_position:
                    stats['too_early'] += 1
                elif answer_pos > self.max_answer_position:
                    stats['too_late'] += 1
                else:
                    # 位置合适，保留这个样本
                    stats['position_ok'] += 1
                    item['answer_position'] = answer_pos
                    item['answer_text'] = answer
                    filtered_data.append(item)
        
        # 打印统计信息
        logger.info("\n答案筛选统计:")
        logger.info(f"  原始样本数: {stats['total']}")
        logger.info(f"  答案在文本中: {stats['has_answer']} ({stats['has_answer']/stats['total']*100:.1f}%)")
        logger.info(f"  答案位置太早 (<{self.min_answer_position*100:.0f}%): {stats['too_early']}")
        logger.info(f"  答案位置太晚 (>{self.max_answer_position*100:.0f}%): {stats['too_late']}")
        logger.info(f"  答案位置合适: {stats['position_ok']} ({stats['position_ok']/stats['total']*100:.1f}%)")
        logger.info(f"  答案未找到: {stats['not_found']}")
        
        if len(filtered_data) == 0:
            logger.warning("\n⚠️  警告：筛选后没有任何样本！将使用原始数据")
            return data
        
        logger.info(f"\n✓ 成功筛选出 {len(filtered_data)} 个合格样本")
        
        # 打印答案位置分布
        if filtered_data:
            positions = [item['answer_position'] for item in filtered_data]
            logger.info(f"\n答案位置分布:")
            logger.info(f"  最小: {min(positions):.2f}")
            logger.info(f"  最大: {max(positions):.2f}")
            logger.info(f"  平均: {sum(positions)/len(positions):.2f}")
            logger.info(f"  中位数: {sorted(positions)[len(positions)//2]:.2f}")
        
        logger.info("=" * 60)
        
        return filtered_data


class MixedLoader(DatasetLoader):
    """混合数据集加载器：结合SQuAD（短文本）和NarrativeQA（长文本）"""
    
    def __init__(self, data_dir: str = "data", max_samples: Optional[int] = None, 
                 random_seed: int = 42, squad_ratio: float = 0.5):
        """
        初始化混合数据集加载器
        
        Args:
            data_dir: 数据存储目录
            max_samples: 最大采样数量（None表示使用默认1000：500 SQuAD + 500 NarrativeQA）
            random_seed: 随机种子
            squad_ratio: SQuAD数据的比例（默认0.5，即各500题）
        """
        super().__init__(data_dir, max_samples, random_seed)
        self.dataset_name = "mixed"
        self.squad_ratio = squad_ratio
        
    def load(self) -> List[Dict]:
        """
        加载混合数据集
        
        Returns:
            混合后的数据集列表
        """
        logger.info("="*60)
        logger.info("加载混合数据集（Mixed Dataset）")
        logger.info("="*60)
        
        # 确定每个数据集的样本数
        if self.max_samples is None:
            # 默认：500 SQuAD + 500 NarrativeQA = 1000
            squad_samples = 500
            narrativeqa_samples = 500
        else:
            squad_samples = int(self.max_samples * self.squad_ratio)
            narrativeqa_samples = self.max_samples - squad_samples
        
        logger.info(f"目标配比：")
        logger.info(f"  - SQuAD (短文本): {squad_samples} 题")
        logger.info(f"  - NarrativeQA (长文本): {narrativeqa_samples} 题")
        logger.info(f"  - 总计: {squad_samples + narrativeqa_samples} 题")
        
        # 加载SQuAD数据
        logger.info(f"\n[1/2] 加载SQuAD数据...")
        squad_loader = SQuADLoader(
            data_dir=self.data_dir,
            max_samples=squad_samples,
            random_seed=self.random_seed
        )
        squad_data = squad_loader.load()
        
        # 给每个样本添加来源标记
        for item in squad_data:
            item['source_dataset'] = 'squad'
            item['id'] = f"squad_{item.get('id', 'unknown')}"
        
        logger.info(f"✓ 加载了 {len(squad_data)} 题 SQuAD")
        if squad_data:
            avg_len = sum(len(item.get('context', '')) for item in squad_data) / len(squad_data)
            logger.info(f"  平均长度: {avg_len:.0f} 字符")
        
        # 加载NarrativeQA数据
        logger.info(f"\n[2/2] 加载NarrativeQA数据...")
        narrativeqa_loader = NarrativeQALoader(
            data_dir=self.data_dir,
            max_samples=narrativeqa_samples,
            random_seed=self.random_seed
        )
        narrativeqa_data = narrativeqa_loader.load()
        
        # 给每个样本添加来源标记
        for item in narrativeqa_data:
            item['source_dataset'] = 'narrativeqa'
            if 'id' in item:
                item['id'] = f"narrativeqa_{item['id']}"
        
        logger.info(f"✓ 加载了 {len(narrativeqa_data)} 题 NarrativeQA")
        if narrativeqa_data:
            avg_len = sum(len(item.get('context', '')) for item in narrativeqa_data) / len(narrativeqa_data)
            logger.info(f"  平均长度: {avg_len:.0f} 字符")
        
        # 混合数据
        mixed_data = squad_data + narrativeqa_data
        
        # 打乱顺序
        random.seed(self.random_seed)
        random.shuffle(mixed_data)
        random.seed()  # 重置
        
        logger.info(f"\n{'='*60}")
        logger.info(f"✓ 混合数据集创建完成")
        logger.info(f"  总样本数: {len(mixed_data)}")
        logger.info(f"  SQuAD: {len(squad_data)} 题 ({len(squad_data)/len(mixed_data)*100:.1f}%)")
        logger.info(f"  NarrativeQA: {len(narrativeqa_data)} 题 ({len(narrativeqa_data)/len(mixed_data)*100:.1f}%)")
        
        # 统计平均长度
        if mixed_data:
            avg_len = sum(len(item.get('context', '')) for item in mixed_data) / len(mixed_data)
            logger.info(f"  整体平均长度: {avg_len:.0f} 字符 (约 {avg_len/4:.0f} tokens)")
        
        logger.info(f"{'='*60}")
        
        return mixed_data


class TriviaQALoader(DatasetLoader):
    """TriviaQA数据集加载器（长上下文问答数据集）"""
    
    def __init__(self, data_dir: str = "data", max_samples: Optional[int] = None, random_seed: int = 42):
        super().__init__(data_dir, max_samples, random_seed)
        self.dataset_name = "triviaqa"
    
    def load(self) -> List[Dict]:
        """
        加载TriviaQA数据集（长上下文数据集）
        
        TriviaQA特点：
        - 平均上下文长度：3000-8000 tokens
        - 任务类型：基于多文档的问答
        - 适合观察长上下文下的性能变化
        
        Returns:
            数据集列表
        """
        dataset_path = self.data_dir / "triviaqa.json"
        
        # 如果文件不存在，尝试从Hugging Face下载
        if not dataset_path.exists():
            logger.info(f"数据集文件不存在，尝试从Hugging Face下载...")
            if self._download_from_huggingface(dataset_path):
                logger.info("数据集下载成功")
            else:
                raise FileNotFoundError(
                    f"无法加载TriviaQA数据集。\n"
                    f"请手动下载数据集并保存到: {dataset_path}\n"
                    f"或安装datasets库: pip install datasets"
                )
        
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 标准化数据格式
            data = self._normalize_data(data)
            
            # 过滤掉上下文太短的样本
            data = [item for item in data if len(item.get("context", "")) > 800]
            logger.info(f"过滤后剩余 {len(data)} 条长上下文数据")
            
            # 采样
            data = self._sample_data(data)
            
            logger.info(f"成功加载 {len(data)} 条数据")
            if data:
                avg_len = sum(len(item.get("context", "")) for item in data) / len(data)
                logger.info(f"平均上下文长度: {avg_len:.0f} 字符 (约 {avg_len/4:.0f} tokens)")
            
            return data
        except Exception as e:
            logger.error(f"加载数据集失败: {e}")
            raise
    
    def _download_from_huggingface(self, save_path: Path) -> bool:
        """
        从Hugging Face下载TriviaQA数据集
        
        Args:
            save_path: 保存路径
            
        Returns:
            是否下载成功
        """
        if not HAS_DATASETS:
            logger.error("需要安装datasets库: pip install datasets")
            return False
        
        try:
            logger.info("正在从Hugging Face下载TriviaQA数据集（这可能需要一些时间，数据集较大）...")
            # 下载TriviaQA数据集（使用rc格式，包含上下文）
            try:
                dataset = load_dataset("trivia_qa", "rc", split="train")
            except:
                # 如果rc格式不可用，尝试其他格式
                dataset = load_dataset("trivia_qa", split="train")
            
            # 转换为标准格式
            data = []
            for item in dataset:
                # TriviaQA的数据结构
                question = item.get("question", "")
                answer = item.get("answer", {})
                
                # 处理答案
                if isinstance(answer, dict):
                    answer_value = answer.get("value", answer.get("normalized_value", ""))
                elif isinstance(answer, str):
                    answer_value = answer
                else:
                    answer_value = ""
                
                # 获取上下文（可能是多个文档）
                search_results = item.get("search_results", [])
                if not search_results:
                    # 如果没有search_results，尝试其他字段
                    search_results = item.get("evidence", [])
                
                # 合并所有上下文文档
                contexts = []
                if isinstance(search_results, list):
                    for result in search_results:
                        if isinstance(result, dict):
                            context_text = result.get("document", {}).get("plain_text", "") or result.get("text", "")
                            if context_text:
                                contexts.append(context_text)
                        elif isinstance(result, str):
                            contexts.append(result)
                
                if not contexts:
                    # 如果还是没有上下文，跳过
                    continue
                
                context = "\n\n".join(contexts)
                
                if context and question and answer_value:
                    data.append({
                        "context": context,
                        "question": question,
                        "answers": [answer_value],
                        "id": item.get("question_id", item.get("id", f"triviaqa_{len(data)}"))
                    })
            
            # 保存到本地
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"下载完成，共 {len(data)} 条数据")
            return True
        except Exception as e:
            logger.error(f"从Hugging Face下载数据集失败: {e}")
            return False
    
    def _normalize_data(self, data: List[Dict]) -> List[Dict]:
        """
        标准化数据格式
        
        Args:
            data: 原始数据
            
        Returns:
            标准化后的数据
        """
        normalized = []
        
        for item in data:
            if isinstance(item, dict):
                context = item.get("context", "")
                question = item.get("question", "")
                answers = item.get("answers", [])
                
                if isinstance(answers, str):
                    answers = [answers]
                
                if context and question and answers:
                    normalized.append({
                        "context": context,
                        "question": question,
                        "answers": answers if answers else [""],
                        "id": item.get("id", f"triviaqa_{len(normalized)}")
                    })
        
        return normalized


def get_dataset_loader(dataset_name: str, 
                      data_dir: str = "data",
                      max_samples: Optional[int] = None,
                      random_seed: int = 42) -> DatasetLoader:
    """
    获取数据集加载器
    
    Args:
        dataset_name: 数据集名称（支持: squad, narrativeqa, triviaqa）
        data_dir: 数据目录
        max_samples: 最大采样数量（None表示使用全部数据，建议500）
        random_seed: 随机种子，用于可复现的采样
        
    Returns:
        数据集加载器实例
    """
    loaders = {
        "squad": SQuADLoader,
        "narrativeqa": NarrativeQALoader,
        "mixed": MixedLoader,
        "triviaqa": TriviaQALoader,
    }
    
    if dataset_name not in loaders:
        raise ValueError(f"未知的数据集: {dataset_name}。支持的数据集: {', '.join(loaders.keys())}")
    
    return loaders[dataset_name](data_dir, max_samples=max_samples, random_seed=random_seed)


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    loader = SQuADLoader()
    data = loader.load()
    print(f"加载了 {len(data)} 条数据")
    if data:
        print(f"第一条数据: {data[0]['question']}")

