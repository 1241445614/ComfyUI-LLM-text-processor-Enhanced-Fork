<!-- README-I18N:START -->
[English](./README.md) | **中文**
<!-- README-I18N:END -->

# ComfyUI-LLM-text-processor (增强分支)

基于 [KingManiya/ComfyUI-LLM-text-processor](https://github.com/KingManiya/ComfyUI-LLM-text-processor) 的增强版本，专注于 **低显存 / 高内存** 场景的本地 GGUF 模型推理优化，新增多项实用功能与前端交互改进。

## ✨ 新增功能

### 🎛️ 性能预设（一键配置）
节点内置 5 种性能预设，选择后自动填充下方所有相关参数，无需手动逐项调整：

| 预设 | 适用场景 | 关键配置 |
|------|----------|----------|
| `balanced` | 均衡模式 | 默认参数，不覆盖用户设置 |
| `low_vram_high_ram` | MoE 模型，低显存高内存 | 专家层全部卸载至 CPU，KV 缓存 `q8_0` |
| `dense_27b_low_vram` | **27B 稠密模型，12G 显存推荐** | 所有层上 GPU，自动附加 `ffn_0_45` 卸载 |
| `hybrid_offload` | MoE 混合卸载 | 部分层 GPU + 专家层 CPU |
| `extreme_low_vram` | 极低显存（≤4GB） | KV 缓存 `q4_0`，最大程度压缩显存 |

### 📦 FFN 张量卸载（独立选项）
通过 `ffn_offload` 下拉菜单控制 `--override-tensor`，将指定层的 FFN 张量卸载到 CPU 内存：

- `none`：不卸载
- `ffn_0_15` / `ffn_0_30` / `ffn_0_45` / `ffn_0_60`：卸载 0~N 层 FFN
- `ffn_0_45_attn`：卸载 0~45 层 FFN + Attention

### 🖼️ 链式图片输入（无需手动设置数量）
- 默认只显示 `image_1` 输入口
- 连上 `image_1` 后自动出现 `image_2`
- 连上 `image_2` 后自动出现 `image_3`，以此类推
- 最多支持 10 个图片/视频帧输入
- **IMAGE batch 自动展开**：视频抽帧后的 batch 会被自动拆分为多帧送入模型

### 🧠 思考模式改进（`reasoning_effort`）
将原先的 `reasoning` 改为 `reasoning_effort` 并合并为单个下拉菜单：

| 选项 | 效果 |
|------|------|
| `off` | 完全关闭思考，响应最快 |
| `low` | 快速推理，输出思考内容 |
| `medium` | 平衡速度与深度（推荐默认） |
| `xhigh` | 深度思考，适用于复杂任务 |

### 📝 系统提示词双输入方式
- `system_prompt`：从预设文件下拉选择
- `system_prompt_text`：直接在节点文本框输入多行提示词
- **优先级**：文本框非空时覆盖文件选择

### 🔧 其他改进
- **KV 缓存量化**：支持 `f16` / `q8_0` / `q4_0` 等类型，`q8_0` 在几乎无损下节省约 47% 显存
- **内存映射控制**：`no_mmap` 可禁用内存映射，配合大内存预加载模型
- **KV 缓存 RAM 上限**：`cache_ram` 限制主机内存中 KV 缓存大小

## 📦 安装

1. 将本仓库克隆到 ComfyUI 的 `custom_nodes` 目录：
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/你的用户名/ComfyUI-LLM-text-processor.git
   ```

2. 安装依赖（如果尚未安装）：
   ```bash
   python -m pip install -r requirements.txt
   ```

3. 重启 ComfyUI，在节点菜单中找到 `LLM Text Processor`。

## 🚀 快速上手（12GB 显存 + 64GB 内存）

1. 将 `performance_preset` 设为 **`dense_27b_low_vram`**
2. 确认 `ffn_offload` 自动变为 `ffn_0_45`
3. 根据任务调整 `ctx_size`（建议 32768 起步）
4. 在 `prompt` 中输入提示词，运行即可

## 📋 参数说明

### 核心参数
| 参数 | 说明 |
|------|------|
| `model` | GGUF 模型文件（从 `ComfyUI/models/LLM` 加载） |
| `mmproj` | 视觉投影文件，连接图片时必选 |
| `system_prompt` | 系统提示词预设文件 |
| `system_prompt_text` | 直接输入系统提示词（优先于文件） |
| `prompt` | 用户提示词 |
| `performance_preset` | 性能预设（自动填充下方参数） |
| `ffn_offload` | FFN 张量卸载层范围 |
| `ctx_size` | 上下文窗口大小 |
| `max_tokens` | 最大生成 token 数 |
| `reasoning_effort` | 思考模式与强度 |

### 高级参数（Advanced）
| 参数 | 说明 |
|------|------|
| `memory_mode` | 显存/内存放置策略 |
| `n_gpu_layers` | GPU 层数 |
| `n_cpu_moe_layers` | CPU MoE 层数 |
| `cache_type_k/v` | KV 缓存量化类型 |
| `no_mmap` | 禁用内存映射 |
| `cache_ram` | KV 缓存 RAM 上限（MiB） |

## 🧪 使用 MTP 加速（可选）

如果你的模型文件名包含 `MTP` 标识（如 `Qwen3.8-27B-...-MTP.gguf`），可以在 `extra_args` 中添加以下参数启用多 Token 预测加速：

```
--spec-type draft-mtp --spec-draft-n-max 3
```

## 📁 文件结构

```
ComfyUI-LLM-text-processor/
├── __init__.py
├── nodes.py              # 节点定义与参数收集
├── llama_cli.py          # 命令行构建与输出解析
├── folder_registry.py    # 模型/提示词文件扫描
├── llama_binary.py       # llama.cpp 二进制管理
├── web/
│   └── llm_text_processor.js  # 前端交互（链式输入、预设联动）
└── README.md
```

## 🙏 致谢

- 原项目：[KingManiya/ComfyUI-LLM-text-processor](https://github.com/KingManiya/ComfyUI-LLM-text-processor)
- [llama.cpp](https://github.com/ggml-org/llama.cpp) 提供底层推理支持

## 📄 许可证

遵循原项目许可证。请查阅原仓库获取详细信息。
