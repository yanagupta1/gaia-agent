import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen


DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"
DEFAULT_GAIA_DATASET_URL = (
    "https://huggingface.co/datasets/gaia-benchmark/GAIA/resolve/main/2023/validation"
)


@dataclass(frozen=True)
class AttachmentContext:
    task_id: str
    file_name: str


class AttachmentTool:
    def __init__(
        self,
        api_url: str | None = None,
        cache_dir: str | None = None,
        timeout_seconds: int | None = None,
    ):
        self.api_url = (api_url or os.getenv("GAIA_API_URL") or DEFAULT_API_URL).rstrip("/")
        self.cache_dir = Path(cache_dir or os.getenv("ATTACHMENT_CACHE_DIR", ".gaia_files"))
        self.timeout_seconds = timeout_seconds or int(os.getenv("ATTACHMENT_TIMEOUT_SECONDS", "30"))
        self.trust_python_attachments = (
            os.getenv("TRUST_GAIA_PYTHON_ATTACHMENTS", "false").lower() == "true"
        )
        self.dataset_url = os.getenv("GAIA_DATASET_FILES_URL", DEFAULT_GAIA_DATASET_URL).rstrip("/")

    def summary(self, context: AttachmentContext) -> str:
        path = self.download(context).resolve()
        return (
            f"Attachment path: {path}\n"
            f"File name: {context.file_name}\n"
            f"Extension: {path.suffix.lower() or '(none)'}\n"
            f"Size bytes: {path.stat().st_size}"
        )

    def read_text(self, context: AttachmentContext, max_chars: int = 12000) -> str:
        path = self.download(context).resolve()
        suffix = path.suffix.lower()
        if suffix in {".txt", ".py", ".csv", ".json", ".md"}:
            return path.read_text(encoding="utf-8", errors="replace")[:max_chars]
        return f"Cannot read {suffix} attachment as plain text."

    def execute_python(self, context: AttachmentContext) -> str:
        """
        Trusted benchmark-only executor.

        This intentionally refuses to run by default because executing a downloaded
        Python file on the host is unsafe outside a trusted benchmark setting. For
        GAIA course runs, enable it explicitly with:

            TRUST_GAIA_PYTHON_ATTACHMENTS=true

        Longer term this should run inside an isolated sandbox/container and return
        only captured stdout/stderr.
        """
        if not self.trust_python_attachments:
            return (
                "Tool error: Python attachment execution is disabled. "
                "This is a trusted benchmark-only executor; set "
                "TRUST_GAIA_PYTHON_ATTACHMENTS=true to enable it."
            )

        path = self.download(context).resolve()
        if path.suffix.lower() != ".py":
            return "Tool error: attachment is not a Python file."
        completed = subprocess.run(
            ["python", str(path)],
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            cwd=str(path.parent),
        )
        output = completed.stdout.strip()
        error = completed.stderr.strip()
        return (
            f"Return code: {completed.returncode}\n"
            f"STDOUT:\n{output}\n"
            f"STDERR:\n{error}"
        ).strip()

    def run_python(self, code: str) -> str:
        """
        General-purpose code interpreter: run short Python the model itself wrote
        and return stdout/stderr. This lets the agent compute math, logic, set, and
        counting answers deterministically instead of guessing. Gated behind the
        same trust flag as attachment execution, since it runs on the host.
        """
        if not self.trust_python_attachments:
            return (
                "Tool error: python execution is disabled. "
                "Set TRUST_GAIA_PYTHON_ATTACHMENTS=true to enable it."
            )
        if not code or not code.strip():
            return "Tool error: no code provided."
        try:
            completed = subprocess.run(
                ["python", "-c", code],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return f"Tool error: code timed out after {self.timeout_seconds}s."
        output = completed.stdout.strip()
        error = completed.stderr.strip()
        return (
            f"Return code: {completed.returncode}\n"
            f"STDOUT:\n{output}\n"
            f"STDERR:\n{error}"
        ).strip()

    def analyze_spreadsheet(self, context: AttachmentContext) -> str:
        import pandas as pd

        path = self.download(context)
        suffix = path.suffix.lower()
        if suffix not in {".xlsx", ".xls", ".csv", ".tsv"}:
            return "Tool error: attachment is not a supported spreadsheet file."

        if suffix == ".csv":
            sheets = {"csv": pd.read_csv(path)}
        elif suffix == ".tsv":
            sheets = {"tsv": pd.read_csv(path, sep="\t")}
        else:
            sheets = pd.read_excel(path, sheet_name=None)

        blocks = []
        for sheet_name, frame in sheets.items():
            displayed = frame if len(frame) <= 200 else frame.head(50)
            rows_label = "All rows" if len(frame) <= 200 else "First 50 rows"
            preview = displayed.to_string(index=False)
            numeric_sums = frame.select_dtypes(include="number").sum(numeric_only=True)
            sums_text = (
                numeric_sums.to_string()
                if not numeric_sums.empty
                else "No numeric columns detected."
            )
            categorical_blocks = []
            for column in frame.select_dtypes(exclude="number").columns:
                values = frame[column].dropna().astype(str).value_counts().head(30)
                if not values.empty:
                    categorical_blocks.append(f"{column}:\n{values.to_string()}")
            categorical_text = (
                "\n\n".join(categorical_blocks)
                if categorical_blocks
                else "No non-numeric columns detected."
            )
            blocks.append(
                f"Sheet: {sheet_name}\n"
                f"Rows: {len(frame)}\n"
                f"Columns: {list(frame.columns)}\n"
                f"Numeric column sums:\n{sums_text}\n"
                f"Non-numeric value counts:\n{categorical_text}\n"
                f"{rows_label}:\n{preview}"
            )
        return "\n\n".join(blocks)

    def download(self, context: AttachmentContext) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / context.file_name
        if path.exists():
            return path

        data = self._fetch_scoring_api(context) or self._fetch_gaia_dataset(context)
        if data is None:
            raise RuntimeError(
                f"Could not download file for task {context.task_id}. The scoring "
                "API has no file for it; set HF_TOKEN (with gated-repo read) so it "
                "can be fetched from the gaia-benchmark/GAIA dataset."
            )
        path.write_bytes(data)
        return path

    def _fetch_scoring_api(self, context: AttachmentContext) -> bytes | None:
        url = f"{self.api_url}/files/{context.task_id}"
        request = Request(url, headers={"User-Agent": "gaia-agent/0.6"})
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return response.read()
        except Exception:
            return None

    def _fetch_gaia_dataset(self, context: AttachmentContext) -> bytes | None:
        token = os.getenv("HF_TOKEN", "").strip()
        if not token or not context.file_name:
            return None
        url = f"{self.dataset_url}/{quote(context.file_name)}"
        request = Request(
            url,
            headers={"Authorization": f"Bearer {token}", "User-Agent": "gaia-agent/0.6"},
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return response.read()
        except Exception:
            return None
