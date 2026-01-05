"""
实验配置文件
定义模型、数据集、任务类型和测试点配置
"""

# ========================================
# 实验配置（Experiment Configuration）
# ========================================
# 答案筛选配置
ANSWER_FILTERING_CONFIG = {
    "enable_answer_filtering": True,  # 启用答案筛选
    "min_answer_position": 0.10,
    "max_answer_position": 0.90,
}

# Qwen系列模型的最大上下文长度映射（基于官方信息）
# 参考：https://qwenlm.github.io/zh/blog/qwen2.5-coder-family/
# Qwen2.5-32B: 131,072 tokens (128K)
# Qwen2.5-7B: 131,072 tokens (128K) 
# Qwen2.5-3B: 32,768 tokens (32K)
# Qwen2.5-1.5B: 131,072 tokens (128K)
# Qwen2.5-0.5B: 32,768 tokens (32K) - 特殊版本
QWEN_MODEL_CONTEXT_LENGTHS = {
    # Qwen2.5系列
    "qwen2.5:32b": 131072,  # 128K
    "qwen2.5:7b": 131072,   # 128K
    "qwen2.5:3b": 32768,    # 32K
    "qwen2.5:1.5b": 131072, # 128K
    "qwen2.5:0.5b": 32768,  # 32K (特殊版本)
    # Qwen2系列（32K上下文）
    "qwen2:72b": 32768,
    "qwen2:57b": 32768,
    "qwen2:32b": 32768,
    "qwen2:14b": 32768,
    "qwen2:7b": 32768,
    "qwen2:3b": 32768,
    "qwen2:1.5b": 32768,
    "qwen2:0.5b": 32768,
    # Qwen系列（8K上下文）
    "qwen:72b": 8192,
    "qwen:32b": 8192,
    "qwen:14b": 8192,
    "qwen:7b": 8192,
    "qwen:4b": 8192,
    "qwen:1.8b": 8192,
    # Qwen1.5系列（32K上下文）
    "qwen1.5:72b": 32768,
    "qwen1.5:32b": 32768,
    "qwen1.5:14b": 32768,
    "qwen1.5:7b": 32768,
    "qwen1.5:4b": 32768,
    "qwen1.5:3b": 32768,
    "qwen1.5:1.8b": 32768,
    "qwen1.5:0.5b": 32768,
}


def get_model_max_context(model_name: str, default: int = 131072) -> int:
    """
    根据模型名称获取最大上下文长度（基于Hugging Face官方信息）
    
    Args:
        model_name: Ollama模型名称（如 "qwen2.5:32b"）
        default: 如果找不到模型时的默认值（默认131072，即128K）
        
    Returns:
        最大上下文长度（token数）
    """
    # 直接查找映射表
    if model_name in QWEN_MODEL_CONTEXT_LENGTHS:
        return QWEN_MODEL_CONTEXT_LENGTHS[model_name]
    
    # 尝试模糊匹配（处理可能的变体）
    model_lower = model_name.lower()
    
    # Qwen2.5系列匹配（注意0.5B和3B版本是32K，其他是128K）
    if "qwen2.5" in model_lower or "qwen2_5" in model_lower:
        if "0.5b" in model_lower or "3b" in model_lower:
            return 32768  # 0.5B和3B版本是32K
        return 131072  # 其他Qwen2.5版本是128K
    
    # Qwen2系列匹配（32K上下文）
    if "qwen2" in model_lower and "qwen2.5" not in model_lower:
        return 32768
    
    # Qwen1.5系列匹配（32K上下文）
    if "qwen1.5" in model_lower or "qwen1_5" in model_lower:
        return 32768
    
    # Qwen系列匹配（旧版本，8K上下文）
    if model_lower.startswith("qwen:") or model_lower.startswith("qwen"):
        return 8192
    
    # 如果都匹配不到，返回默认值
    return default


# Qwen模型配置（只测试这3个模型）
QWEN_MODELS = {
    "qwen2.5-32b": {
        "ollama_name": "qwen2.5:32b",
        "vllm_name": "qwen2.5-32b",
        "params": "32B",
        "max_context": None,  # 将自动从get_model_max_context获取
        "expected_threshold": 0.25,  # 预期临界点25%
        "dir_name": "qwen2.5-32b"  # 目录名
    },
    "qwen2.5-7b": {
        "ollama_name": "qwen2.5:7b",
        "vllm_name": "qwen2.5-7b",
        "params": "7B",
        "max_context": None,  # 将自动从get_model_max_context获取
        "expected_threshold": 0.25,
        "dir_name": "qwen2.5-7b"
    },
    "qwen2.5-3b": {
        "ollama_name": "qwen2.5:3b",
        "vllm_name": "qwen2.5-3b",
        "params": "3B",
        "max_context": None,  # 将自动从get_model_max_context获取
        "expected_threshold": 0.25,
        "dir_name": "qwen2.5-3b"
    }
}

# 任务类型到目录名的映射
TASK_TYPE_TO_DIR = {
    "information_retrieval": "information-retrieval",
    "reading_comprehension": "reading-comprehension",
    "logical_reasoning": "logic-reasoning",
    "math_calculation": "math-calculation"
}

# 上下文长度测试点配置
CONTEXT_LENGTH_POINTS = {
    "fine_grained": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
    "coarse_grained": [0.60, 0.70, 0.80, 0.90, 1.00],
    "all": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
            0.60, 0.70, 0.80, 0.90, 1.00],
    "ten_percent": [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],  # 10%间隔，共9个测试点
    # 密集测试点（用于断崖优化，test_round3）
    "dense": [
        0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
        0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95
    ]  # 19个点，每5%
}

# 任务类型配置
TASK_TYPES = {
    "information_retrieval": {
        "name": "信息检索",
        "metrics": ["recall", "precision", "mrr"],
        "description": "在长文本中查找特定信息"
    },
    "reading_comprehension": {
        "name": "阅读理解",
        "metrics": ["f1"],  # 只保留F1，移除accuracy和exact_match
        "description": "回答关于文本的问题"
    },
    "logical_reasoning": {
        "name": "逻辑推理",
        "metrics": ["accuracy", "reasoning_steps_correct"],
        "description": "基于长文本进行多步推理"
    },
    "math_calculation": {
        "name": "数学计算",
        "metrics": ["accuracy", "calculation_correct"],
        "description": "在长文本中提取数字并进行计算"
    },
}

# 数据集配置
DATASETS = {
    "squad": {
        "name": "SQuAD",
        "type": "reading_comprehension",
        "description": "标准阅读理解数据集（短上下文，平均约500 tokens）",
        "hf_dataset": "squad",
        "download_url": "https://huggingface.co/datasets/squad",
        "avg_context_length": "~500 tokens"
    },
    "narrativeqa": {
        "name": "NarrativeQA",
        "type": "reading_comprehension",
        "description": "长上下文阅读理解数据集（基于完整书籍/剧本，平均6000-15000 tokens）",
        "hf_dataset": "narrativeqa",
        "download_url": "https://huggingface.co/datasets/narrativeqa",
        "avg_context_length": "6000-15000 tokens",
        "recommended": True,  # 推荐用于长上下文实验
        "supported_tasks": ["reading_comprehension"]  # 仅支持阅读理解任务
    },
    "mixed": {
        "name": "Mixed Dataset",
        "type": "mixed",
        "description": "混合数据集：500题SQuAD（短文本）+ 500题NarrativeQA（长文本）",
        "composition": {"squad": 500, "narrativeqa": 500},
        "avg_context_length": "mixed (500-50000 tokens)",
        "recommended": True,  # 推荐用于全面评估
        "supported_tasks": ["reading_comprehension"]
    },
    "triviaqa": {
        "name": "TriviaQA",
        "type": "reading_comprehension",
        "description": "长上下文问答数据集（多文档证据，平均3000-8000 tokens）",
        "hf_dataset": "trivia_qa",
        "download_url": "https://huggingface.co/datasets/trivia_qa",
        "avg_context_length": "3000-8000 tokens",
        "recommended": True  # 推荐用于长上下文实验
    }
}

# 实验参数配置
EXPERIMENT_CONFIG = {
    "temperature": 0.0,  # 固定温度，确保确定性输出
    "random_seed": 42,   # 固定随机种子
    "num_repeats": 1,    # 每个配置重复1次
    "max_tokens": 512,   # 最大生成token数
    "timeout": 300,      # 请求超时时间（秒）
    "batch_size": 1,    # 批处理大小
    "dataset_max_samples": 500,  # 每个数据集的最大采样数量（None表示使用全部数据）
    # 答案筛选配置
    "filter_samples_with_answers": ANSWER_FILTERING_CONFIG.get("enable_answer_filtering", False),
    "min_answer_position": ANSWER_FILTERING_CONFIG.get("min_answer_position", 0.10),
    "max_answer_position": ANSWER_FILTERING_CONFIG.get("max_answer_position", 0.90),
}

# Ollama配置
OLLAMA_CONFIG = {
    "base_url": "http://localhost:11434",
    "api_endpoint": "/api/generate",
    "timeout": 300
}

# vLLM配置
VLLM_CONFIG = {
    "base_url": "http://localhost:11434/v1",
    "api_key": None,
    "timeout": 300
}

# 结果保存配置
RESULTS_CONFIG = {
    "results_dir": "results",
    "logs_dir": "logs",
    "plots_dir": "plots",
    "checkpoint_interval": 10  # 每10个实验保存一次检查点
}

# 临界点判定阈值
THRESHOLD_CONFIG = {
    "mild_degradation": 0.10,   # 10%下降为轻微下降
    "significant_degradation": 0.20,  # 20%下降为显著下降
    "cliff_degradation": 0.30,  # 30%下降为断崖式下降（临界点）
    "statistical_significance": 0.05  # 统计显著性水平
}

