import os
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from file_tools import AttachmentContext, AttachmentTool


class LocalAttachmentTool(AttachmentTool):
    def __init__(self, local_path: Path):
        super().__init__(cache_dir=str(local_path.parent))
        self.local_path = local_path

    def download(self, context):
        return self.local_path


class FileToolsTests(unittest.TestCase):
    def setUp(self):
        self.test_root = Path(".test_tmp")
        self.test_root.mkdir(exist_ok=True)

    def tearDown(self):
        if self.test_root.exists():
            shutil.rmtree(self.test_root, ignore_errors=True)

    def test_python_execution_disabled_by_default(self):
        directory = self.test_root / "disabled"
        directory.mkdir()
        path = directory / "code.py"
        path.write_text("print(123)", encoding="utf-8")
        with patch.dict(os.environ, {}, clear=True):
            tool = LocalAttachmentTool(path)
            result = tool.execute_python(AttachmentContext("task", "code.py"))
        self.assertIn("disabled", result)
        self.assertIn("TRUST_GAIA_PYTHON_ATTACHMENTS=true", result)

    def test_python_execution_requires_explicit_trust(self):
        directory = self.test_root / "trusted"
        directory.mkdir()
        path = directory / "code.py"
        path.write_text("print(123)", encoding="utf-8")
        with patch.dict(
            os.environ,
            {"TRUST_GAIA_PYTHON_ATTACHMENTS": "true"},
            clear=True,
        ):
            tool = LocalAttachmentTool(path)
            result = tool.execute_python(AttachmentContext("task", "code.py"))
        self.assertIn("Return code: 0", result)
        self.assertIn("123", result)


if __name__ == "__main__":
    unittest.main()
