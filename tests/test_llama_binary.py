from __future__ import annotations

import tempfile
import sys
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import llama_binary

class ExistingInstallTests(unittest.TestCase):
    def test_only_current_release_is_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            vendor_root = Path(temp)
            for tag in ("old-release", llama_binary.LLAMA_CPP_RELEASE_TAG):
                install_dir = vendor_root / tag / llama_binary.WINDOWS_CUDA_13.key
                install_dir.mkdir(parents=True)
                for name in llama_binary.WINDOWS_CUDA_13.required_files:
                    (install_dir / name).touch()

            with mock.patch.object(llama_binary, "VENDOR_ROOT", vendor_root):
                paths = llama_binary._existing_install(llama_binary.WINDOWS_CUDA_13)

            self.assertEqual(
                paths.cli,
                vendor_root
                / llama_binary.LLAMA_CPP_RELEASE_TAG
                / llama_binary.WINDOWS_CUDA_13.key
                / llama_binary.WINDOWS_CUDA_13.cli_executable,
            )


if __name__ == "__main__":
    unittest.main()
