# 快速开始：微调Qwen2.5-7B缓解断崖式降智

## 前置要求

### 1. 硬件要求
- **GPU**: 至少40GB显存（推荐A100 40GB或更高）
- **内存**: 至少64GB RAM
- **磁盘**: 至少50GB可用空间（用于模型和数据集）

### 2. 软件要求
```bash
# 安装依赖
pip install --default-timeout=6000 -r requirements.txt
pip install flash-attn --no-build-isolation
```

## 步骤1：下载模型到本地（推荐）

### 为什么先下载？

1. **避免重复下载**：模型约14GB，下载一次后可以重复使用
2. **离线使用**：下载后可以离线微调
3. **速度更快**：本地加载比每次从HuggingFace下载快得多

### 下载命令

```bash
python finetune_download_model.py \
    --model-name Qwen/Qwen2.5-7B-Instruct \
    --local-dir models/qwen2.5-7b-instruct
```

**参数说明**：
- `--model-name`: HuggingFace模型名称
- `--local-dir`: 本地保存目录（默认：`models/qwen2.5-7b-instruct`）

**下载时间**：
- 取决于网络速度
- 7B模型约14GB，通常需要30分钟到2小时

**下载位置**：
- 模型会保存到 `models/qwen2.5-7b-instruct/` 目录
- 包含：`config.json`, `tokenizer.json`, `model-*.safetensors` 等文件

## 步骤2：准备数据集

```bash
python finetune_prepare_dataset.py \
    --input-dir data \
    --output-file data_finetune/finetune_dataset.json \
    --max-context-length 131072 \
    --cliff-start 0.40 \
    --cliff-end 0.50 \
    --augmentation-ratio 2.0
```

这会：
1. 加载SQuAD和NarrativeQA数据
2. 识别临界区域（40-50%）的样本
3. 进行数据增强（段落重排、上下文切片等）
4. 保存到 `data_finetune/finetune_dataset.json`

## 步骤3：运行微调

### 方式1：使用本地模型（推荐）

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=4,5 torchrun \
    --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29502 \
    finetune_cliff_mitigation.py \
      --model /data/models/Qwen/Qwen2.5-7B-Instruct \
      --data-dir data_finetune \
      --output-dir finetune_output \
      --batch-size 1 \
      --learning-rate 2e-5 \
      --num-epochs 3 \
      --critical-weight 3.0 \
      --enable-rope-tuning \
      --enable-data-aug \
      --augmentation-ratio 2.0 \
      --deepspeed ds_zero3_config.json \
      --grad-accum-steps 8 \
      --attn-impl flash_attention_2 \
      --max-context-length 32768
```

### 方式2：直接从HuggingFace下载（不推荐）

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=4,5 torchrun \
    --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29502 \
    finetune_cliff_mitigation.py \
      --model Qwen/Qwen2.5-7B-Instruct \
      --data-dir data_finetune \
      --output-dir finetune_output \
      --batch-size 1 \
      --learning-rate 2e-5 \
      --num-epochs 3 \
      --critical-weight 3.0 \
      --enable-rope-tuning \
      --enable-data-aug \
      --augmentation-ratio 2.0 \
      --deepspeed ds_zero3_config.json \
      --grad-accum-steps 8 \
      --attn-impl flash_attention_2 \
      --max-context-length 32768
```

**注意**：如果网络不稳定，建议先下载到本地。

## 完整流程示例

```bash
# 1. 下载模型（首次运行）
python finetune_download_model.py \
    --model-name Qwen/Qwen2.5-7B-Instruct \
    --local-dir models/qwen2.5-7b-instruct

# 2. 准备数据集
python finetune_prepare_dataset.py \
    --input-dir data \
    --output-file data_finetune/finetune_dataset.json \
    --max-context-length 131072 \
    --cliff-start 0.40 \
    --cliff-end 0.50 \
    --augmentation-ratio 2.0

# 3. 运行微调
PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=4,5 torchrun \
    --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29502 \
    finetune_cliff_mitigation.py \
      --model /data/models/Qwen/Qwen2.5-7B-Instruct \
      --data-dir data_finetune \
      --output-dir finetune_output \
      --batch-size 1 \
      --learning-rate 2e-5 \
      --num-epochs 3 \
      --critical-weight 3.0 \
      --enable-rope-tuning \
      --enable-data-aug \
      --augmentation-ratio 2.0 \
      --deepspeed ds_zero3_config.json \
      --grad-accum-steps 8 \
      --attn-impl flash_attention_2 \
      --max-context-length 32768
```

## 微调参数说明

### 关键参数

- `--model`: 模型路径（本地路径或HuggingFace名称）
- `--critical-weight`: 临界区域权重倍数（3.0表示临界区域样本权重是普通样本的3倍）
- `--enable-rope-tuning`: 启用RoPE参数微调
- `--enable-data-aug`: 启用数据增强
- `--batch-size`: 批次大小（长上下文建议使用1）
- `--learning-rate`: 学习率（建议2e-5到5e-5）
- `--num-epochs`: 训练轮数（建议2-5轮）

### 内存优化参数

如果GPU内存不足，可以调整：

```bash
# 使用更小的batch size和更多的gradient accumulation
--batch-size 1 \
--gradient-accumulation-steps 16  # 等效batch_size=16

# 或者减小max_context_length（如果可能）
# 需要修改代码中的max_context_length参数
```

## 监控训练过程

训练过程中会输出：
- 训练损失（加权损失）
- 验证损失
- 学习率变化
- 训练进度

**关键指标**：
- `train_loss`: 应该逐渐下降
- `eval_loss`: 应该跟随train_loss下降
- 如果eval_loss开始上升，可能过拟合，考虑早停

## 保存的模型

微调完成后，模型会保存到 `finetune_output/` 目录：
- `config.json`: 模型配置
- `tokenizer.json`: tokenizer文件
- `model-*.safetensors`: 模型权重
- `training_args.bin`: 训练参数

## 使用微调后的模型

### 方法1：使用transformers库

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "finetune_output"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
```

### 方法2：评估微调效果

```bash
# 使用微调后的模型进行测试
python main_natural.py --dataset mixed --model [微调后的模型名] --task reading_comprehension --max-samples 1000 --llm-backend vllm --vllm-url xxxxx --vllm-api-key xxxxx

# 检测断崖点位置
python detect_cliff_point.py --results-dir results/mixed --model [微调后的模型名] --dataset mixed --task reading_comprehension --metric f1

# 对比微调前后的断崖点位置
```

## 常见问题

### Q1: 模型下载失败怎么办？

**A**: 
1. 检查网络连接
2. 使用镜像站点（如果在中国）：
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   ```
3. 手动下载：访问 https://huggingface.co/Qwen/Qwen2.5-7B-Instruct

### Q2: GPU内存不足怎么办？

**A**:
1. 减小batch_size（使用1）
2. 增加gradient_accumulation_steps
3. 使用gradient checkpointing（需要修改代码）
4. 减小max_context_length（如果可能）

### Q3: 训练很慢怎么办？

**A**:
1. 使用更强大的GPU（A100或更高）
2. 减小数据量（先在小数据集上测试）
3. 使用混合精度训练（已启用bf16）

### Q4: 如何知道微调是否有效？

**A**:
1. 监控训练损失是否下降
2. 使用detect_cliff_point.py检测断崖点位置
3. 对比微调前后的性能曲线

## 下一步

微调完成后：
1. 评估微调效果（使用detect_cliff_point.py）
2. 对比微调前后的断崖点位置
3. 分析性能改善程度
4. 如果效果不理想，调整参数重新微调

## 参考

- 详细文档：`finetune_README.md`
- 数据集准备：`finetune_prepare_dataset.py`
- 微调脚本：`finetune_cliff_mitigation.py`

