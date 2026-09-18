<!-- README-I18N:START -->
**English** | [中文](./README.zh.md)
<!-- README-I18N:END -->
# ComfyUI-LLM-text-processor (Enhanced Fork)

An enhanced version of [KingManiya/ComfyUI-LLM-text-processor](https://github.com/KingManiya/ComfyUI-LLM-text-processor), focused on optimizing local GGUF model inference for **low VRAM / high RAM** scenarios, with several new practical features and frontend interaction improvements.

## ✨ New Features

### 🎛️ Performance Presets (One-Click Configuration)
The node includes 5 built-in performance presets. Once selected, all related parameters below are filled in automatically, so you don't need to adjust them one by one:

| Preset | Use Case | Key Configuration |
|--------|----------|-------------------|
| `balanced` | Balanced mode | Default parameters, does not override user settings |
| `low_vram_high_ram` | MoE models, low VRAM & high RAM | All expert layers offloaded to CPU, KV cache `q8_0` |
| `dense_27b_low_vram` | **27B dense models, recommended for 12GB VRAM** | All layers on GPU, automatically adds `ffn_0_45` offload |
| `hybrid_offload` | MoE hybrid offload | Some layers on GPU + expert layers on CPU |
| `extreme_low_vram` | Extremely low VRAM (≤4GB) | KV cache `q4_0`, maximum VRAM compression |

### 📦 FFN Tensor Offloading (Standalone Option)
Use the `ffn_offload` dropdown to control `--override-tensor` and offload FFN tensors from specified layers to CPU memory:

- `none`: No offloading
- `ffn_0_15` / `ffn_0_30` / `ffn_0_45` / `ffn_0_60`: Offload FFN layers 0 to N
- `ffn_0_45_attn`: Offload FFN + Attention layers 0 to 45

### 🖼️ Chained Image Inputs (No Manual Count Setting Needed)
- Only `image_1` is shown by default
- After connecting `image_1`, `image_2` appears automatically
- After connecting `image_2`, `image_3` appears, and so on
- Supports up to 10 image/video frame inputs
- **Automatic IMAGE batch expansion**: batches from video frame extraction are automatically split into multiple frames and sent to the model

### 🧠 Better reasoning mode (`reasoning_effort`)
The old `reasoning` has been replaced with the `reasoning_effort` option and combined into a dropdown:

| Option | Effect |
|--------|--------|
| `off` | Completely disables reasoning, fastest response |
| `low` | Fast reasoning, outputs reasoning content |
| `medium` | Balanced speed and depth (recommended default) |
| `xhigh` | Deep reasoning, suitable for complex tasks |

### 📝 Dual System Prompt Input
- `system_prompt`: Select from preset files via dropdown
- `system_prompt_text`: Directly enter multi-line prompts in the node text box
- **Priority**: When the text box is non-empty, it overrides the file selection

### 🔧 Other Improvements
- **KV Cache Quantization**: Supports `f16` / `q8_0` / `q4_0` and more. `q8_0` saves about 47% VRAM with almost no quality loss
- **Memory Mapping Control**: `no_mmap` can disable memory mapping and preload models with large RAM
- **KV Cache RAM Limit**: `cache_ram` limits the KV cache size in host memory

## 📦 Installation

1. Clone this repository into ComfyUI's `custom_nodes` directory:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/your-username/ComfyUI-LLM-text-processor.git
   ```

2. Install dependencies (if not already installed):
   ```bash
   python -m pip install -r requirements.txt
   ```

3. Restart ComfyUI and find `LLM Text Processor` in the node menu.

## 🚀 Quick Start (12GB VRAM + 64GB RAM)

1. Set `performance_preset` to **`dense_27b_low_vram`**
2. Confirm that `ffn_offload` automatically becomes `ffn_0_45`
3. Adjust `ctx_size` as needed (recommended starting point: 32768)

## 📋 Parameter Reference

### Core Parameters
| Parameter | Description |
|-----------|-------------|
| `model` | GGUF model file (loaded from `ComfyUI/models/LLM`) |
| `mmproj` | Vision projector file, required when connecting images |
| `system_prompt` | System prompt preset file |
| `system_prompt_text` | Directly enter system prompt (overrides file) |
| `prompt` | User prompt |
| `performance_preset` | Performance preset (auto-fills parameters below) |
| `ffn_offload` | FFN tensor offload layer range |
| `ctx_size` | Context window size |
| `max_tokens` | Maximum number of tokens to generate |
| `reasoning_effort` | Reasoning mode and intensity |

### Advanced Parameters
| Parameter | Description |
|-----------|-------------|
| `memory_mode` | VRAM/RAM placement strategy |
| `n_gpu_layers` | Number of GPU layers |
| `n_cpu_moe_layers` | Number of CPU MoE layers |
| `cache_type_k/v` | KV cache quantization type |
| `no_mmap` | Disable memory mapping |
| `cache_ram` | KV cache RAM limit (MiB) |

## 🧪 Using MTP Acceleration (Optional)

If your model filename contains the `MTP` identifier (e.g. `Qwen3.8-27B-...-MTP.gguf`), you can add the following parameters to `extra_args` to enable Multi-Token Prediction acceleration:

```
--spec-type draft-mtp --spec-draft-n-max 3
```

## 📁 File Structure

```
ComfyUI-LLM-text-processor/
├── __init__.py
├── nodes.py              # Node definition and parameter collection
├── llama_cli.py          # Command construction and output parsing
├── folder_registry.py    # Model/prompt file scanning
├── llama_binary.py       # llama.cpp binary management
├── web/
│   └── llm_text_processor.js  # Frontend interaction (chained inputs, preset linkage)
└── README.md
```

## 🙏 Acknowledgements

- Original project: [KingManiya/ComfyUI-LLM-text-processor](https://github.com/KingManiya/ComfyUI-LLM-text-processor)
- [llama.cpp](https://github.com/ggml-org/llama.cpp) provides the underlying inference support

## 📄 License

Follows the original project's license. Please refer to the original repository for details.
