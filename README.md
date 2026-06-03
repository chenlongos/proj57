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

## 评价标准

最终奖项由技术委员会评审专家综合评定。评审按以下优先级排序：

### 第一优先级：推理正确性

这是最基本也是最重要的要求。权重最高。

- 在目标平台上完成完整的推理流水线：图像预处理 → 模型推理 → 动作后处理
- 输出动作的**决策方向与参考一致**（如差速小车左转/右转的判断不能反向）
- 允许因量化、剪枝等优化引入数值偏差，不要求 bit-exact 精确匹配

### 第二优先级：工程实现质量

同等正确性的情况下，以下维度作为区分依据：

- **资源利用效率**：针对平台资源限制（尤其是内存）做了有效优化，如量化、剪枝、内存复用等
- **推理速度**：在正确性不受损的前提下，推理越快越好
- **运行稳定性**：长时间运行不出错，是工程质量的体现
- **代码质量**：代码结构清晰、可读性好、注释完善
- **文档**：部署流程说明清晰，评委和后续开发者可复现，要求包含可执行文件大小 占用内存情况 执行推理时长，和左转右转的标志性数据
- **AI使用说明**: 需要在文档中公开自己的工作研究和开发过程，包括是否借鉴其他人的代码，是否借助了AI进行了各种程度的开发，写清AI参与和人工参与的部分

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

## 常见问题（FAQ）

### Q1: "跑通 ACT 推理"的评估维度是什么？是否会考察实时性、内存占用、稳定性等指标？

比赛的首要目标是**推理结果的正确性**——即预处理、模型推理、后处理流水线完整跑通，输出的动作数据在合理误差范围内，动作趋势正确。评审专家组在综合评定时会额外关注工程实现质量（内存优化、代码质量、平台适配等），但**不设硬性的实时性指标**（如最低 FPS 或最大延迟阈值），也没有严格的稳定性压力测试要求。

### Q2: 剪枝和量化后，模型输出与原始 FP32 模型一样吗？

**不一样。** 剪枝和量化都会引入数值偏差，不可能是 bit-exact 完全一致：

- **剪枝**：直接移除权重或神经元，改变计算图结构
- **量化**（FP32→INT8/FP16）：每层计算都会产生舍入误差并累积

评判推理正确性时**不要求输出数值精确一致**，而是考察**决策方向是否一致**。以差速驱动小车为例：

- 原模型判定左转（`left_vel < right_vel`） → 优化后仍判定左转 ✅
- 原模型判定左转 → 优化后变成右转 ❌（方向反转，不合格）

选手在模型优化后应自行验证输出动作的符号和相对大小关系是否与原始模型一致。

### Q3: SG2002（256MB）内存不足时，是否允许使用 mmap、外部存储、swap 等方式？

**允许。** SG2002 仅 256MB 内存，是比赛的核心挑战。选手可自由选择合理的技术手段来满足内存约束：

- **mmap**：允许。将模型权重文件通过内存映射直接访问
- **外部存储（TF 卡）**：允许。模型文件、输入数据可存放在外部存储上
- **TPU SDK 自带的量化和压缩工具**：允许且鼓励使用
- **系统 swap**：不禁止，但存储 I/O 速度有限，过度依赖可能导致推理极慢

总体原则：在给定硬件上，选手可自由选择技术手段，只要推理结果正确即可。

### Q4: 推理程序是否允许长期占用大部分系统资源？

**允许。** StarryOS 作为轻量级嵌入式操作系统，核心服务占用资源极少。推理任务为本比赛的核心负载，推理程序可以充分利用 CPU/TPU/内存资源，无需为其他非必要后台服务预留配额。
