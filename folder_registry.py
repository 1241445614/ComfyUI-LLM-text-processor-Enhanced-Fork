from __future__ import annotations

import os
from pathlib import Path

import folder_paths


LLM_FOLDER = "llm_text_processor_models"
PROMPT_FOLDER = "llm_text_processor_prompts"
NO_SYSTEM_PROMPT = "none"
NO_MMPROJ = "none"
NO_MTP = "none"
NO_MODELS_FOUND = "No GGUF models found"

# MTP 草稿头专用子目录（位于 ComfyUI/models/LLM/ 下）
MTP_DRAFT_SUBDIR = "MTP_draft"


def llm_root() -> Path:
    return Path(folder_paths.models_dir) / "LLM"


def prompt_root() -> Path:
    return llm_root() / "prompts"


def mtp_draft_root() -> Path:
    return llm_root() / MTP_DRAFT_SUBDIR


def register_folders() -> None:
    llm_dir = llm_root()
    prompts_dir = prompt_root()
    mtp_dir = mtp_draft_root()

    llm_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)
    mtp_dir.mkdir(parents=True, exist_ok=True)

    folder_paths.folder_names_and_paths[LLM_FOLDER] = ([str(llm_dir)], {".gguf"})
    folder_paths.folder_names_and_paths[PROMPT_FOLDER] = ([str(prompts_dir)], {".txt"})


# ---------------------------------------------------------------------------
# 文件分类辅助
# ---------------------------------------------------------------------------

def _is_mmproj(name: str) -> bool:
    return "mmproj" in Path(name).name.lower()


def _is_in_mtp_draft(name: str) -> bool:
    """判断文件路径是否位于 MTP_draft 子目录下。

    name 是相对于 LLM 根目录的相对路径，例如：
      - "Qwen3.8/Huihui-...-mtp.gguf"     -> False
      - "MTP_draft/mtp-gemma-4-E4B-it.gguf" -> True
    """
    normalized = Path(name).as_posix()
    return normalized.startswith(MTP_DRAFT_SUBDIR + "/")


# ---------------------------------------------------------------------------
# 下拉列表：模型 / mmproj / MTP 草稿头 / 系统提示词
# ---------------------------------------------------------------------------

def model_options() -> list[str]:
    """主模型下拉列表。

    包含：
      - 普通主模型
      - 内嵌 MTP 的主模型
    排除：
      - mmproj 文件
      - MTP_draft 子目录下的所有文件（避免污染主模型列表）
    """
    files = folder_paths.get_filename_list(LLM_FOLDER)
    models = [
        name
        for name in files
        if not _is_mmproj(name) and not _is_in_mtp_draft(name)
    ]
    return models or [NO_MODELS_FOUND]


def mmproj_options() -> list[str]:
    files = folder_paths.get_filename_list(LLM_FOLDER)
    mmproj = [name for name in files if _is_mmproj(name)]
    return [NO_MMPROJ] + mmproj


def mtp_options() -> list[str]:
    """MTP 草稿头下拉列表。

    只读取 MTP_draft 子目录下的 GGUF 文件，其他目录一律忽略。
    """
    files = folder_paths.get_filename_list(LLM_FOLDER)
    mtp_files = [
        name
        for name in files
        if _is_in_mtp_draft(name) and not _is_mmproj(name)
    ]
    return [NO_MTP] + mtp_files


def system_prompt_options() -> list[str]:
    files = folder_paths.get_filename_list(PROMPT_FOLDER)
    top_level_files = [
        name for name in files if os.sep not in name and "/" not in name
    ]
    return [NO_SYSTEM_PROMPT] + top_level_files


# ---------------------------------------------------------------------------
# 路径解析
# ---------------------------------------------------------------------------

def full_model_path(name: str) -> Path:
    if name == NO_MODELS_FOUND:
        raise FileNotFoundError(
            f"No GGUF model files were found in {llm_root()}. "
            f"Place a .gguf model there and refresh ComfyUI."
        )
    path = folder_paths.get_full_path(LLM_FOLDER, name)
    if path is None:
        raise FileNotFoundError(f"GGUF model not found: {name}")
    return Path(path)


def full_mmproj_path(name: str) -> Path | None:
    if name == NO_MMPROJ:
        return None
    path = folder_paths.get_full_path(LLM_FOLDER, name)
    if path is None:
        raise FileNotFoundError(f"mmproj GGUF file not found: {name}")
    return Path(path)


def full_mtp_path(name: str) -> Path | None:
    if name == NO_MTP:
        return None
    path = folder_paths.get_full_path(LLM_FOLDER, name)
    if path is None:
        raise FileNotFoundError(f"MTP draft GGUF file not found: {name}")
    return Path(path)


def full_system_prompt_path(name: str) -> Path | None:
    if name == NO_SYSTEM_PROMPT:
        return None
    path = folder_paths.get_full_path(PROMPT_FOLDER, name)
    if path is None:
        raise FileNotFoundError(f"System prompt preset not found: {name}")
    return Path(path)


register_folders()
