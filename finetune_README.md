# 断崖式降智缓解微调指南

## 概述

本目录包含用于缓解LLM断崖式降智现象的微调实现，主要包括：

1. **临界区域加权损失**：对40-50%临界区域的样本给予更高权重
2. **RoPE参数微调**：改善位置编码的外推能力
3. **临界区域数据增强**：增加临界区域的训练样本

## 文件说明

- `finetune_cliff_mitigation.py` - 主微调脚本
- `finetune_prepare_dataset.py` - 数据集准备脚本（临界区域数据增强）

## 快速开始

### 1. 准备数据集

首先运行数据集准备脚本，对临界区域进行数据增强：

```bash
python finetune_prepare_dataset.py \
    --input-dir data \
    --output-file data_finetune/finetune_dataset.json \
    --max-context-length 131072 \
    --cliff-start 0.40 \
    --cliff-end 0.50 \
    --augmentation-ratio 2.0
```

**参数说明**：
- `--input-dir`: 输入数据目录（包含squad.json和narrativeqa.json）
- `--output-file`: 输出增强后的数据集路径
- `--max-context-length`: 最大上下文长度（128K = 131072）
- `--cliff-start`: 临界区域起始比率（0.40 = 40%）
- `--cliff-end`: 临界区域结束比率（0.50 = 50%）
- `--augmentation-ratio`: 增强倍数（2.0表示临界区域样本翻倍）

### 2. 运行微调

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

**参数说明**：
- `--model`: 模型名称（HuggingFace格式）
- `--data-dir`: 数据目录
- `--output-dir`: 微调后的模型保存目录
- `--batch-size`: 批次大小（长上下文建议使用1）
- `--learning-rate`: 学习率
- `--num-epochs`: 训练轮数
- `--critical-weight`: 临界区域权重倍数（3.0表示临界区域样本的损失权重是普通样本的3倍）
- `--enable-rope-tuning`: 启用RoPE参数微调
- `--enable-data-aug`: 启用数据增强
- `--augmentation-ratio`: 数据增强倍数

## 数据集构造原理

### 临界区域数据增强策略

1. **段落重排**：
   - 将长文本按段落分割
   - 随机重排段落顺序
   - 保持文本长度在临界区域（40-50%）

2. **上下文切片**：
   - 从长文本中截取不同部分
   - 确保截取后的长度在临界区域
   - 生成多个变体

3. **样本复制**：
   - 如果上述方法不够，复制现有临界区域样本
   - 增加该区域的训练密度

### 数据分布

增强后的数据集分布：
- **其他区域**（<40%或>50%）：保持原始分布
- **临界区域**（40-50%）：增强2-3倍，提高训练密度

## 加权损失函数原理

### 权重计算

```python
if cliff_start <= ratio <= cliff_end:
    weight = critical_weight  # 例如 3.0
else:
    weight = 1.0
```

### 损失计算

```python
# 标准损失
per_sample_loss = cross_entropy_loss(logits, labels)

# 加权损失
weighted_loss = (per_sample_loss * weights).mean()
```

**效果**：临界区域样本的梯度更大，模型在该区域学习更充分。

## RoPE微调原理

### 为什么需要微调RoPE？

- RoPE（Rotary Position Embedding）在超出训练长度时外推能力有限
- 微调RoPE参数可以改善长距离位置编码的外推能力
- 使模型更好地理解长上下文中的位置关系

### 实现方式

1. 找到模型中的RoPE相关层
2. 启用这些参数的梯度
3. 在训练过程中更新这些参数

## 评估微调效果

微调完成后，使用原有的检测脚本评估效果：

```bash
# 1. 使用微调后的模型进行测试
python main_natural.py --dataset mixed --model [微调后的模型名] --task reading_comprehension --max-samples 1000 --llm-backend vllm --vllm-url xxxxx --vllm-api-key xxxxx

# 2. 检测断崖点位置
python detect_cliff_point.py --results-dir results/mixed --model [微调后的模型名] --dataset mixed --task reading_comprehension --metric f1

# 3. 对比微调前后的断崖点位置
```

### 预期效果

- **理想情况**：断崖点推后（从40-50%推到60-70%）或消除
- **现实情况**：下降幅度减小（从45.5%降到20-30%）

## 注意事项

### 1. 内存要求

- 长上下文微调需要大量GPU内存
- 建议使用：
  - `batch_size=1`
  - `gradient_accumulation_steps=8`（等效batch_size=8）
  - `bf16=True`（混合精度训练）

### 2. 计算资源

- 128K上下文长度的微调需要：
  - GPU内存：至少40GB（A100）
  - 训练时间：根据数据量，可能需要数小时到数天

### 3. 灾难性遗忘

- 微调可能提升长上下文性能，但降低短上下文性能
- 建议：
  - 使用多任务学习（同时优化所有长度区间）
  - 定期评估短上下文性能
  - 保存最佳checkpoint

### 4. 过拟合风险

- 在临界区域过度拟合可能导致泛化能力下降
- 建议：
  - 使用验证集监控性能
  - 早停机制
  - 交叉验证

## 故障排除

### 问题1：CUDA out of memory

**解决方案**：
- 减小`batch_size`（使用1）
- 增加`gradient_accumulation_steps`
- 使用`gradient_checkpointing`
- 减小`max_context_length`（如果可能）

### 问题2：找不到RoPE层

**解决方案**：
- 检查模型架构
- 对于某些模型，RoPE可能在embedding层中
- 可以手动指定需要微调的层

### 问题3：数据增强效果不明显

**解决方案**：
- 增加`augmentation_ratio`
- 尝试不同的增强策略
- 检查增强后的数据是否真的在临界区域

## 进阶使用

### 自定义权重函数

可以修改`CriticalRegionDataset`类中的权重计算逻辑：

```python
# 线性权重
weight = 1.0 + (critical_weight - 1.0) * (ratio - cliff_start) / (cliff_end - cliff_start)

# 高斯权重
weight = 1.0 + (critical_weight - 1.0) * np.exp(-((ratio - center)**2) / (2 * sigma**2))
```

### 课程学习

可以实现渐进式长度训练：

```python
# 阶段1：短上下文（<30%）
# 阶段2：中长上下文（30-50%）
# 阶段3：长上下文（>50%）
```

### 多任务学习

同时优化多个长度区间：

```python
loss = loss_short + loss_medium + loss_long
```

## 引用

如果使用本微调方法，请引用：

```
断崖式降智缓解微调方法
- 临界区域加权损失
- RoPE参数微调
- 临界区域数据增强
```

## 联系方式

如有问题或建议，请提交Issue或Pull Request。

