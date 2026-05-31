# Pro57 — ACT 模型嵌入式推理部署

## 赛题背景

本赛题旨在推动国产操作系统 **StarryOS** 与前沿具身智能技术 **ACT 模型**的深度融合。通过在国产化嵌入式平台上部署与优化模仿学习模型，验证系统对复杂 AI 任务的支持能力，促进自主可控的智能生态建设。

**ACT (Action Chunking Transformer)** 是一种基于 Transformer 编码器-解码器架构的行为克隆模型，论文 [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://huggingface.co/papers/2304.13705)。模型输入图像和机器人状态，输出多步未来动作预测。

## 赛题任务

使用提供的统一 ACT 模型，在 StarryOS 上完成 ACT 推理，输出预期数据。

| 任务 | 平台 | 内存    | 推理后端 | 语言 |
|------|------|-------|----------|------|
| 任务一 | 荔枝派 SG2002 | 256MB | TPU | C/C++/Rust |
| 任务二 | 香橙派 RK3588 | 4/8GB | NPU | C/C++/Rust |
| 任务三 | QEMU（StarryOS） | —     | CPU | C/C++/Rust |

## 难度分层

<!-- 以下为题目本身完成难度的分析，并非最终评奖标准。最终奖项由技术委员会评审专家综合评定。 -->

| 等级 | 要求 | 说明 |
|------|------|------|
| **基础** | 任务三 | QEMU 环境，CPU 上运行 ACT 推理 |
| **进阶** | 任务三 + 任务二 | RK3588 NPU 推理 |
| **优秀** | 任务一 | SG2002 256MB TPU 推理（资源极度受限） |
| **卓越** | 任务一 + 任务二 | 多平台覆盖 + 良好的工程实现与优化 |

## 组织方提供

本仓库提供以下可直接使用的资源：

```
pro57/
├── act/                          # ACT 模型定义（PyTorch）
│   ├── modeling_act.py           # 完整模型架构（含 CVAE、TemporalEnsembler）
│   ├── configuration_act.py      # 模型配置
│   ├── defaults.py               # 默认配置和工厂函数
│   └── ACTDataset.py             # 数据加载和预处理
├── train.ipynb                   # 训练 Notebook
├── infer.ipynb                   # PC 端推理 Notebook（参考实现）
├── download_data.ipynb           # 模型和数据下载
└── output/                   # 示例数据
    ├── dataset/ # 训练数据集
    │   ├── data/                  # Parquet 数据文件
    │   ├── meta/stats.json        # 归一化统计量（QUANTILES）
    │   └── videos/                # JPEG 图像帧 (320×240)
    └── train/
        └── model.pt              # 预训练模型 checkpoint
```

### PC 端快速体验

```bash
# 安装依赖
pip install torch torchvision numpy pandas Pillow pyarrow jupyter matplotlib

# 启动 Notebook
jupyter notebook train.ipynb   # 训练
jupyter notebook infer.ipynb   # 推理
```

## 选手需要自行完成

以下内容均由选手自行实现：

1. **模型导出** — PyTorch → ONNX（或其他中间格式）
2. **模型优化**（可选）— 量化（INT8/FP16）、剪枝、知识蒸馏
3. **平台适配** — 根据目标平台选择推理框架：
   - QEMU: ONNX Runtime
   - RK3588: RKNN Toolkit2（ONNX → RKNN）
   - SG2002: Sophon TPU SDK（需重点优化内存占用）
4. **StarryOS 交叉编译** — 配置交叉编译工具链，编写 CMake/Makefile
5. **C/C++/Rust 推理程序** — 实现图像预处理、模型推理、动作后处理

## 模型规格

选手需要了解以下关键规格以完成导出和部署。

### 输入

| 名称 | Shape | 说明 |
|------|-------|------|
| `image` | `[1, 3, 224, 224]` | RGB 图像，需从原始 320×240 resize 到 224×224 |
| `state` | `[1, 2]` | 机器人状态 `[left_vel, right_vel]`，使用 QUANTILES 归一化 |
| `latent` | `[1, 32]` | CVAE 隐变量（推理时从训练统计中采样的固定分布） |

**预处理流程**:
1. 图像: `Resize(224,224)` → `ToTensor()` → `Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])`
2. 状态: `2 * (x - q01) / (q99 - q01) - 1`（QUANTILES 归一化，参数见 `meta/stats.json`）

### 输出

| 名称 | Shape | 说明 |
|------|-------|------|
| `action` | `[1, action_chunk_size, action_dim]` | 预测的动作 chunk（默认 8 步 × 3 维） |

**后处理流程**:
- 反归一化: `(action + 1) / 2 * (q99 - q01) + q01`
- 输出维度: `[left_vel, right_vel, gripper_target]`
- 实际执行时通常只取第一个时间步

### 模型架构

```
Image (224×224×3) ──► RGBEncoder (ResNet18) ──► [B, tokens, 512]
State [vel_l, vel_r] ──► StateEncoder (MLP)   ──► [B, 1, 512]
CVAE latent z ──► LatentProj ──► [B, 1, 512]
                                      │
                              Concat + PosEmbed
                                      │
                         Transformer Encoder (4 layers)
                                      │
                         Transformer Decoder (4 layers) ◄── learned queries
                                      │
                         ActionHead ──► [B, 8, 3]
```

### 关键参数（默认）

| 参数 | 值 | 说明 |
|------|-----|------|
| `state_dim` | 2 | 状态维度 |
| `action_dim` | 3 | 动作维度 |
| `action_chunk_size` | 8 | 一次预测 8 步 |
| `hidden_dim` | 512 | Transformer 隐藏维度 |
| `dim_feedforward` | 3200 | FFN 维度 |
| `num_encoder_layers` | 4 | Encoder 层数 |
| `num_decoder_layers` | 4 | Decoder 层数 |
| `num_attention_heads` | 8 | 注意力头数 |
| `latent_dim` | 32 | CVAE 隐变量维度 |
| `use_cvae` | True | 启用 CVAE |

## 模型 checkpoint 格式

```python
checkpoint = {
    "model_state_dict": OrderedDict,          # PyTorch 权重
    "config": {                               # 模型配置字典
        "state_dim": 2, "action_dim": 3,
        "action_chunk_size": 8, "hidden_dim": 512, ...
    },
    "inference_latent_mu": Tensor[32],        # CVAE 推理均值
    "inference_latent_log_sigma": Tensor[32], # CVAE 推理 log 方差
}
```

## 数据集格式

### 目录结构

```
dataset/<user_id>/default/
├── data/chunk-000/file-*.parquet   # 数据（包含 observation.image/state/action 列）
├── meta/
│   ├── stats.json                  # QUANTILES 归一化参数
│   └── info.json                   # 数据集元信息
└── videos/observation.images.fpv/chunk-000/frame_*.jpg  # JPEG 图像帧
```

### stats.json 示例

```json
{
  "observation.state": { "q01": [-0.023, -0.100], "q99": [0.200, 0.200] },
  "action": { "q01": [-0.100, -0.100, 0.0], "q99": [0.200, 0.200, 0.0] }
}
```

## 参考资源

- StarryOS: https://github.com/Starry-OS/StarryOS
- 荔枝派 SG2002: https://wiki.sipeed.com/hardware/zh/lichee/RV_Nano/5_peripheral.html
- RK3588 RKNN Toolkit2: https://github.com/airockchip/rknn-toolkit2
- ACT 论文: [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://huggingface.co/papers/2304.13705)
