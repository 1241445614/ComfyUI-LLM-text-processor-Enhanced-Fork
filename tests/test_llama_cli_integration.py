from __future__ import annotations

import importlib
import itertools
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
IMAGE_PATH = PROJECT_ROOT / "images" / "node.png"
ENV_PATH = Path(__file__).with_name(".env")

import llama_binary


def load_test_env() -> None:
    if not ENV_PATH.exists():
        return
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.strip().partition("=")
        if separator and key and not key.startswith("#"):
            os.environ.setdefault(key, value.strip().strip("\"'"))


load_test_env()


def import_llama_cli():
    model_management = types.ModuleType("comfy.model_management")
    model_management.processing_interrupted = lambda: False
    model_management.throw_exception_if_processing_interrupted = lambda: None
    comfy = types.ModuleType("comfy")
    comfy.model_management = model_management
    package = types.ModuleType("llm_text_processor_testpkg")
    package.__path__ = [str(PROJECT_ROOT)]

    with mock.patch.dict(sys.modules, {
        "comfy": comfy,
        "comfy.model_management": model_management,
        "llm_text_processor_testpkg": package,
        "llm_text_processor_testpkg.llama_binary": llama_binary,
    }):
        return importlib.import_module("llm_text_processor_testpkg.llama_cli")


class ImageTensor:
    def __init__(self, path: Path):
        self.array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255

    def dim(self):
        return self.array.ndim

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


@unittest.skipUnless(
    all(os.environ.get(name) for name in (
        "LLAMA_INTEGRATION_CLI",
        "LLAMA_INTEGRATION_MODEL",
        "LLAMA_INTEGRATION_MMPROJ",
    )),
    "copy tests/.env.example to tests/.env and set the integration paths",
)
class LlamaCliIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cli = Path(os.environ["LLAMA_INTEGRATION_CLI"])
        cls.model = Path(os.environ["LLAMA_INTEGRATION_MODEL"])
        cls.mmproj = Path(os.environ["LLAMA_INTEGRATION_MMPROJ"])
        cls.llama_cli = import_llama_cli()
        cls.image = ImageTensor(IMAGE_PATH)
        cls.system_prompt = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", encoding="utf-8", delete=False
        )
        cls.system_prompt.write("Be concise.")
        cls.system_prompt.close()
        cls.system_prompt_path = Path(cls.system_prompt.name)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.system_prompt_path.unlink(missing_ok=True)

    def test_all_input_reasoning_and_output_limit_combinations(self) -> None:
        for system, image, reasoning, fits in itertools.product(
            (False, True), (False, True), ("off", "on"), (False, True)
        ):
            with self.subTest(system=system, image=image, reasoning=reasoning, fits=fits):
                prompt = (
                    "Say OK."
                    if fits
                    else "Write BEGIN, then the numbers from 1 to 50, then END."
                )
                with mock.patch.object(
                    self.llama_cli,
                    "ensure_llama_cli_paths",
                    return_value=llama_binary.LlamaCliPaths(cli=self.cli),
                ):
                    command, cleanup_paths = self.llama_cli.build_command(
                        model_path=self.model,
                        mmproj_path=self.mmproj if image else None,
                        system_prompt_path=self.system_prompt_path if system else None,
                        image=self.image if image else None,
                        prompt=prompt,
                        max_tokens=1024 if fits else 8,
                        temperature=0,
                        top_p=1,
                        top_k=1,
                        repeat_penalty=1,
                        ctx_size=2048 if fits else 768,
                        memory_mode="gpu_layers",
                        n_gpu_layers=99,
                        n_cpu_moe_layers=1,
                        seed=42,
                        reasoning=reasoning,
                    )
                response, reasoning_text, perf = self.llama_cli.run_llama_cli(
                    command, timeout_seconds=120, cleanup_paths=cleanup_paths
                )

                self.assertTrue(perf)
                self.assertEqual(bool(reasoning_text), reasoning == "on")
                if fits:
                    self.assertTrue(response.startswith("OK"))
                else:
                    self.assertNotIn("END", response)


if __name__ == "__main__":
    unittest.main()
