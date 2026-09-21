from __future__ import annotations

import os
import re
import shlex
import subprocess
import tempfile
import time
from pathlib import Path

import comfy.model_management

from .llama_binary import ensure_llama_cli_paths

PROMPT_ECHO_END = "... (truncated)"
PROMPT_PADDING = " " * 501
PERF_RE = re.compile(r"\[\s*Prompt:\s*[^|\]]+\|\s*Generation:\s*[^\]]+\]")
MMPROJ_EMBEDDING_MISMATCH_RE = re.compile(
    r"mismatch between text model \(n_embd = (?P<text_embd>\d+)\) and mmproj "
    r"\(n_embd = (?P<mmproj_embd>\d+)\)",
    flags=re.IGNORECASE,
)
START_THINKING = "[Start thinking]"
END_THINKING = "[End thinking]"
LLAMA_RANDOM_SEED = -1
LLAMA_SEED_MODULUS = 2**32
MAX_LLAMA_SEED = LLAMA_SEED_MODULUS - 1

FFN_OFFLOAD_PATTERNS = {
    "ffn_0_15": r"blk\.([0-9]|1[0-5])\.ffn_.*=CPU",
    "ffn_0_30": r"blk\.([0-9]|[1-2][0-9]|30)\.ffn_.*=CPU",
    "ffn_0_45": r"blk\.([0-9]|[1-3][0-9]|4[0-5])\.ffn_.*=CPU",
    "ffn_0_60": r"blk\.([0-9]|[1-5][0-9]|60)\.ffn_.*=CPU",
    "ffn_0_45_attn": r"blk\.([0-9]|[1-3][0-9]|4[0-5])\.(ffn_|attn_).*=CPU",
}


def _tensor_to_temp_png(tensor) -> Path:
    import numpy as np
    from PIL import Image

    array = (tensor.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    pil_image = Image.fromarray(array)
    fd, path = tempfile.mkstemp(prefix="llm-text-processor-", suffix=".png")
    os.close(fd)
    pil_image.save(path, format="PNG")
    return Path(path)


def tensor_to_temp_pngs(image) -> list[Path]:
    if hasattr(image, "dim") and image.dim() == 4:
        return [_tensor_to_temp_png(tensor) for tensor in image]
    return [_tensor_to_temp_png(image)]


def _write_temp_text_file(prefix: str, text: str) -> Path:
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".txt")
    os.close(fd)
    text_path = Path(path)
    text_path.write_text(text, encoding="utf-8", newline="\n")
    return text_path


def _write_prompt_file(prompt: str) -> Path:
    return _write_temp_text_file(
        "llm-text-processor-prompt-", prompt.strip() + PROMPT_PADDING
    )


def split_extra_args(extra_args: str) -> list[str]:
    if not extra_args or not extra_args.strip():
        return []
    parts = shlex.split(extra_args, posix=(os.name != "nt"))
    return [part.strip("\"'") for part in parts]


def normalize_llama_seed(seed: int) -> int:
    seed = int(seed)
    if seed == LLAMA_RANDOM_SEED:
        return LLAMA_RANDOM_SEED
    if 0 <= seed <= MAX_LLAMA_SEED:
        return seed
    return seed % LLAMA_SEED_MODULUS


def _apply_preset(
    performance_preset: str,
    memory_mode: str,
    n_gpu_layers: int,
    n_cpu_moe_layers: int,
    cache_type_k: str,
    cache_type_v: str,
    no_mmap: bool,
    cache_ram: int,
    ffn_offload: str,
) -> tuple:
    if performance_preset == "low_vram_high_ram":
        memory_mode = "cpu_moe_layers"
        n_cpu_moe_layers = 999
        cache_type_k = "q8_0"
        cache_type_v = "q8_0"
        no_mmap = True
        cache_ram = 4096
    elif performance_preset == "dense_27b_low_vram":
        memory_mode = "gpu_layers"
        n_gpu_layers = 999
        n_cpu_moe_layers = 1
        cache_type_k = "q8_0"
        cache_type_v = "q8_0"
        no_mmap = True
        cache_ram = 8192
        if ffn_offload == "none":
            ffn_offload = "ffn_0_45"
    elif performance_preset == "hybrid_offload":
        memory_mode = "gpu_and_cpu_moe_layers"
        n_gpu_layers = 20
        n_cpu_moe_layers = 999
        cache_type_k = "q8_0"
        cache_type_v = "q8_0"
        no_mmap = True
        cache_ram = 4096
    elif performance_preset == "extreme_low_vram":
        memory_mode = "cpu_moe_layers"
        n_cpu_moe_layers = 999
        cache_type_k = "q4_0"
        cache_type_v = "q4_0"
        no_mmap = True
        cache_ram = 2048

    return (
        memory_mode,
        n_gpu_layers,
        n_cpu_moe_layers,
        cache_type_k,
        cache_type_v,
        no_mmap,
        cache_ram,
        ffn_offload,
    )


def build_command(
    model_path: Path,
    mmproj_path: Path | None,
    mtp_draft_path: Path | None,
    system_prompt_path: Path | None,
    system_prompt_text: str,
    images: list,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repeat_penalty: float,
    ctx_size: int,
    performance_preset: str,
    ffn_offload: str,
    mtp_mode: str,
    mtp_n_max: int,
    memory_mode: str,
    n_gpu_layers: int,
    n_cpu_moe_layers: int,
    cache_type_k: str,
    cache_type_v: str,
    no_mmap: bool,
    cache_ram: int,
    seed: int,
    reasoning_effort: str,
    extra_args: list[str] | None = None,
) -> tuple[list[str], tuple[Path | None, ...]]:
    cleanup_paths = []
    cli_paths = ensure_llama_cli_paths()

    image_paths: list[Path] = []
    if images:
        if mmproj_path is None:
            raise ValueError("Image input requires a selected mmproj GGUF file.")
        for img in images:
            image_paths.extend(tensor_to_temp_pngs(img))
        cleanup_paths.extend(image_paths)

    prompt_path = _write_prompt_file(prompt)
    cleanup_paths.append(prompt_path)

    system_prompt_file: Path | None = None
    if system_prompt_text and system_prompt_text.strip():
        system_prompt_file = _write_temp_text_file(
            "llm-text-processor-sysprompt-",
            system_prompt_text.strip(),
        )
        cleanup_paths.append(system_prompt_file)
    elif system_prompt_path is not None:
        system_prompt_file = system_prompt_path

    (
        memory_mode,
        n_gpu_layers,
        n_cpu_moe_layers,
        cache_type_k,
        cache_type_v,
        no_mmap,
        cache_ram,
        ffn_offload,
    ) = _apply_preset(
        performance_preset=performance_preset,
        memory_mode=memory_mode,
        n_gpu_layers=n_gpu_layers,
        n_cpu_moe_layers=n_cpu_moe_layers,
        cache_type_k=cache_type_k,
        cache_type_v=cache_type_v,
        no_mmap=no_mmap,
        cache_ram=cache_ram,
        ffn_offload=ffn_offload,
    )

    command = [
        str(cli_paths.cli),
        "-m",
        str(model_path),
        "-n",
        str(max_tokens),
        "--temp",
        str(temperature),
        "--top-p",
        str(top_p),
        "--top-k",
        str(top_k),
        "--repeat-penalty",
        str(repeat_penalty),
        "-c",
        str(ctx_size),
        "--seed",
        str(normalize_llama_seed(seed)),
        "--single-turn",
    ]

    # 思考模式
    if reasoning_effort == "off":
        command.extend(["--reasoning", "off"])
        command.extend(
            ["--chat-template-kwargs", '{"enable_thinking": false}']
        )
    else:
        command.extend(["--reasoning", "on"])
        command.extend(
            [
                "--chat-template-kwargs",
                f'{{"reasoning_effort": "{reasoning_effort}"}}',
            ]
        )

    # MTP 加速
    if mtp_mode == "on":
        command.extend(["--spec-type", "draft-mtp"])
        command.extend(["--spec-draft-n-max", str(mtp_n_max)])
        # 只有指定了外部草稿头时才传 --spec-draft-model
        # 否则 llama.cpp 会自动使用主模型内嵌的 MTP 头
        if mtp_draft_path is not None:
            command.extend(["--spec-draft-model", str(mtp_draft_path)])

    # KV 缓存量化
    command.extend(["--cache-type-k", cache_type_k])
    command.extend(["--cache-type-v", cache_type_v])

    # 内存映射控制（新版 llama.cpp 使用 --load-mode 代替 --no-mmap）
    if no_mmap:
        command.extend(["--load-mode", "none"])
    if cache_ram > 0:
        command.extend(["--cache-ram", str(cache_ram)])

    # 显存 / 内存放置策略
    if memory_mode in {"gpu_layers", "gpu_and_cpu_moe_layers"}:
        command.extend(["-ngl", str(n_gpu_layers)])
    if memory_mode in {"cpu_moe_layers", "gpu_and_cpu_moe_layers"}:
        command.extend(["--n-cpu-moe", str(n_cpu_moe_layers)])

    # FFN 张量卸载
    if ffn_offload and ffn_offload != "none":
        pattern = FFN_OFFLOAD_PATTERNS.get(ffn_offload)
        if pattern:
            command.extend(["--override-tensor", pattern])

    # Flash Attention
    command.extend(["--flash-attn", "on"])

    if system_prompt_file is not None:
        command.extend(["-sysf", str(system_prompt_file)])

    command.extend(["-f", str(prompt_path)])

    if image_paths:
        command.extend(["--mmproj", str(mmproj_path)])
        command.extend(
            ["--image", ",".join(str(path) for path in image_paths)]
        )

    if extra_args:
        command.extend(extra_args)

    return command, tuple(cleanup_paths)


def run_llama_cli(
    command: list[str],
    timeout_seconds: int,
    cleanup_paths: tuple[Path | None, ...] = (),
) -> tuple[str, str, str]:
    process = None
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
        )
        stdout, stderr = _communicate_with_interrupt(process, timeout_seconds)
        result = subprocess.CompletedProcess(
            command, process.returncode, stdout, stderr
        )
    except BaseException:
        if process is not None:
            _stop_process(process)
        raise
    finally:
        for path in cleanup_paths:
            if path and path.exists():
                path.unlink()

    if result.returncode != 0:
        stderr = result.stderr.strip()
        message = _parse_llama_error(stderr)
        if message:
            raise RuntimeError(message)
        raise RuntimeError(
            f"llama.cpp inference failed with exit code "
            f"{result.returncode}:\n{stderr}"
        )

    return _parse_response(result.stdout + "\n" + result.stderr)


def _stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)


def _communicate_with_interrupt(
    process: subprocess.Popen, timeout_seconds: int
) -> tuple[str, str]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        if comfy.model_management.processing_interrupted():
            _stop_process(process)
            comfy.model_management.throw_exception_if_processing_interrupted()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _stop_process(process)
            raise TimeoutError(
                f"llama.cpp timed out after {timeout_seconds}s"
            )
        try:
            return process.communicate(timeout=min(0.1, remaining))
        except subprocess.TimeoutExpired:
            continue


def _parse_response(text: str) -> tuple[str, str, str]:
    text = str(text or "")

    if PROMPT_ECHO_END in text:
        text = text.split(PROMPT_ECHO_END, 1)[1]

    text = re.sub(r"\n?\s*Exiting\.\.\.\s*$", "", text, flags=re.IGNORECASE)

    perf = ""
    perf_match = PERF_RE.search(text)
    if perf_match:
        perf = perf_match.group(0).strip()
        text = text[: perf_match.start()] + text[perf_match.end():]

    reasoning = ""
    start_idx = text.find(START_THINKING)
    end_idx = text.find(END_THINKING)
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        reasoning = text[start_idx + len(START_THINKING): end_idx].strip()
        text = text[:start_idx] + text[end_idx + len(END_THINKING):]

    response = text.strip()
    return (response, reasoning, perf)


def _parse_llama_error(stderr: str) -> str | None:
    if not stderr:
        return None
    lines = stderr.strip().splitlines()
    for line in lines:
        if "error:" in line.lower():
            return line.strip()
    for line in lines:
        lower = line.lower()
        if "failed" in lower or "out of memory" in lower or "oom" in lower:
            return line.strip()
    return None
