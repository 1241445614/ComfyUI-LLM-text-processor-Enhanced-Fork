from __future__ import annotations

from .folder_registry import (
    NO_MMPROJ,
    full_mmproj_path,
    full_model_path,
    full_system_prompt_path,
    mmproj_options,
    model_options,
    system_prompt_options,
)
from .llama_cli import (
    MAX_LLAMA_SEED,
    build_command,
    run_llama_cli,
    split_extra_args,
)

MAX_IMAGE_INPUTS = 10


class LLMTextProcessor:
    @classmethod
    def INPUT_TYPES(cls):
        # 预定义 image_1 .. image_10，前端按链式自动显示
        image_optional = {}
        for i in range(1, MAX_IMAGE_INPUTS + 1):
            image_optional[f"image_{i}"] = (
                "IMAGE",
                {
                    "tooltip": f"第 {i} 个图片/视频帧输入。"
                    f"连上当前接口后会自动出现下一个。",
                },
            )

        return {
            "required": {
                "model": (
                    model_options(),
                    {
                        "tooltip": "GGUF model loaded from ComfyUI/models/LLM.",
                    },
                ),
                "mmproj": (
                    mmproj_options(),
                    {
                        "default": NO_MMPROJ,
                        "tooltip": "Vision projector GGUF. Required when "
                        "images are connected.",
                    },
                ),
                "system_prompt": (
                    system_prompt_options(),
                    {
                        "tooltip": "从预设文件中选择系统提示词。"
                        "如果 system_prompt_text 非空，则后者优先。",
                    },
                ),
                "system_prompt_text": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": False,
                        "tooltip": "直接输入系统提示词（多行）。"
                        "非空时优先于 system_prompt 文件选择。",
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "default": "Describe this image in detail.",
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": "User prompt sent to the selected model.",
                    },
                ),
                "performance_preset": (
                    [
                        "balanced",
                        "low_vram_high_ram",
                        "dense_27b_low_vram",
                        "hybrid_offload",
                        "extreme_low_vram",
                    ],
                    {
                        "default": "balanced",
                        "tooltip": "性能预设（选择后自动填充下方参数）：\n"
                        "balanced=均衡模式\n"
                        "low_vram_high_ram=低显存高内存（MoE）\n"
                        "dense_27b_low_vram=27B稠密模型专用（12G推荐）\n"
                        "hybrid_offload=混合卸载\n"
                        "extreme_low_vram=极低显存",
                    },
                ),
                "ffn_offload": (
                    [
                        "none",
                        "ffn_0_15",
                        "ffn_0_30",
                        "ffn_0_45",
                        "ffn_0_60",
                        "ffn_0_45_attn",
                    ],
                    {
                        "default": "none",
                        "tooltip": "FFN 张量卸载层范围（--override-tensor）",
                    },
                ),
                "ctx_size": (
                    "INT",
                    {
                        "default": 32768,
                        "min": 512,
                        "max": 1048576,
                        "step": 512,
                        "tooltip": "Context window size in tokens.",
                    },
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": 2048,
                        "min": 1,
                        "max": 32768,
                        "tooltip": "Maximum number of tokens to generate.",
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Sampling temperature.",
                    },
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 0.8,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Nucleus sampling threshold.",
                    },
                ),
                "top_k": (
                    "INT",
                    {
                        "default": 20,
                        "min": 1,
                        "max": 1000,
                        "tooltip": "Top-K sampling cutoff.",
                    },
                ),
                "repeat_penalty": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 3.0,
                        "step": 0.01,
                        "tooltip": "Penalty applied to repeated tokens.",
                    },
                ),
                "reasoning_effort": (
                    ["off", "low", "medium", "xhigh"],
                    {
                        "default": "medium",
                        "tooltip": "思考模式（合并开关与强度）",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 1,
                        "min": -1,
                        "max": MAX_LLAMA_SEED,
                        "tooltip": "Random seed. Use -1 for a random seed.",
                    },
                ),
                "timeout_seconds": (
                    "INT",
                    {
                        "default": 300,
                        "min": 10,
                        "max": 3600,
                        "tooltip": "Maximum time to wait before stopping.",
                    },
                ),
                "memory_mode": (
                    [
                        "auto",
                        "gpu_layers",
                        "cpu_moe_layers",
                        "gpu_and_cpu_moe_layers",
                    ],
                    {
                        "default": "auto",
                        "tooltip": "Advanced memory placement mode.",
                        "advanced": True,
                    },
                ),
                "n_gpu_layers": (
                    "INT",
                    {
                        "default": 99,
                        "min": -1,
                        "max": 999,
                        "tooltip": "Number of model layers on the GPU.",
                        "advanced": True,
                    },
                ),
                "n_cpu_moe_layers": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 999,
                        "tooltip": "Number of MoE layers to keep on the CPU.",
                        "advanced": True,
                    },
                ),
                "cache_type_k": (
                    ["f16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"],
                    {
                        "default": "f16",
                        "tooltip": "KV缓存K量化类型。",
                        "advanced": True,
                    },
                ),
                "cache_type_v": (
                    ["f16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"],
                    {
                        "default": "f16",
                        "tooltip": "KV缓存V量化类型。",
                        "advanced": True,
                    },
                ),
                "no_mmap": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "禁用内存映射，将模型预加载到RAM。",
                        "advanced": True,
                    },
                ),
                "cache_ram": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 65536,
                        "step": 512,
                        "tooltip": "KV缓存RAM回退上限（MiB）。",
                        "advanced": True,
                    },
                ),
            },
            "optional": {
                **image_optional,
                "enable_processing": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "When enabled, run normal node processing.",
                        "advanced": True,
                    },
                ),
                "extra_args": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Optional advanced llama.cpp parameters.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("RESPONSE", "REASONING", "PERF")
    OUTPUT_TOOLTIPS = (
        "Final model response with reasoning blocks removed.",
        "Extracted reasoning when present in model output.",
        "llama.cpp prompt and generation speed.",
    )
    FUNCTION = "generate"
    CATEGORY = "LLM Text Processor"
    TITLE = "LLM Text Processor"

    @classmethod
    def VALIDATE_INPUTS(cls, model, mmproj, system_prompt, **kwargs):
        return True

    def generate(
        self,
        model: str,
        mmproj: str,
        system_prompt: str,
        system_prompt_text: str,
        prompt: str,
        performance_preset: str,
        ffn_offload: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repeat_penalty: float,
        ctx_size: int,
        memory_mode: str,
        n_gpu_layers: int,
        n_cpu_moe_layers: int,
        cache_type_k: str,
        cache_type_v: str,
        no_mmap: bool,
        cache_ram: int,
        seed: int,
        timeout_seconds: int,
        reasoning_effort: str,
        image_1=None,
        image_2=None,
        image_3=None,
        image_4=None,
        image_5=None,
        image_6=None,
        image_7=None,
        image_8=None,
        image_9=None,
        image_10=None,
        enable_processing: bool = True,
        extra_args: str = "",
    ):
        if not enable_processing:
            return (prompt, "", "")

        current_models = model_options()
        current_mmprojs = mmproj_options()
        current_system_prompts = system_prompt_options()

        if model not in current_models:
            raise ValueError(f"Model not found: {model}")
        if mmproj not in current_mmprojs:
            raise ValueError(f"mmproj not found: {mmproj}")
        if system_prompt not in current_system_prompts:
            raise ValueError(
                f"System prompt preset not found: {system_prompt}"
            )

        model_path = full_model_path(model)
        mmproj_path = full_mmproj_path(mmproj)
        if system_prompt_text and system_prompt_text.strip():
            system_prompt_path = None
        else:
            system_prompt_path = full_system_prompt_path(system_prompt)

        # 严格按 image_1..image_10 的顺序收集
        all_images = [
            img
            for img in [
                image_1, image_2, image_3, image_4, image_5,
                image_6, image_7, image_8, image_9, image_10,
            ]
            if img is not None
        ]

        parsed_extra_args = split_extra_args(extra_args)

        command, cleanup_paths = build_command(
            model_path=model_path,
            mmproj_path=mmproj_path,
            system_prompt_path=system_prompt_path,
            system_prompt_text=system_prompt_text,
            images=all_images,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            ctx_size=ctx_size,
            performance_preset=performance_preset,
            ffn_offload=ffn_offload,
            memory_mode=memory_mode,
            n_gpu_layers=n_gpu_layers,
            n_cpu_moe_layers=n_cpu_moe_layers,
            cache_type_k=cache_type_k,
            cache_type_v=cache_type_v,
            no_mmap=no_mmap,
            cache_ram=cache_ram,
            seed=seed,
            reasoning_effort=reasoning_effort,
            extra_args=parsed_extra_args,
        )

        response, reasoning_text, perf = run_llama_cli(
            command=command,
            timeout_seconds=timeout_seconds,
            cleanup_paths=cleanup_paths,
        )
        return (response, reasoning_text, perf)


NODE_CLASS_MAPPINGS = {"LLMTextProcessor": LLMTextProcessor}
NODE_DISPLAY_NAME_MAPPINGS = {"LLMTextProcessor": "LLM Text Processor"}