import argparse
import csv
import json
import os
import time
from pathlib import Path
from urllib.request import urlopen

from agent import GaiaAgent


DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"
DEFAULT_OUTPUT_DIR = "local_runs"


def fetch_questions(api_url: str) -> list[dict]:
    with urlopen(f"{api_url.rstrip('/')}/questions", timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def select_questions(
    questions: list[dict],
    limit: int | None = None,
    task_ids: set[str] | None = None,
) -> list[dict]:
    if task_ids:
        selected = [item for item in questions if item.get("task_id") in task_ids]
    else:
        selected = questions
    if limit is not None:
        selected = selected[:limit]
    return selected


def run_local_eval(
    api_url: str,
    output_dir: Path,
    limit: int | None,
    task_ids: set[str] | None,
) -> tuple[Path, Path, Path]:
    questions = select_questions(fetch_questions(api_url), limit=limit, task_ids=task_ids)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    jsonl_path = output_dir / f"answers-{timestamp}.jsonl"
    csv_path = output_dir / f"answers-{timestamp}.csv"
    report_path = output_dir / f"report-{timestamp}.md"

    agent = GaiaAgent()
    rows = []

    print(f"Running local dry eval on {len(questions)} questions.")
    print("This does not submit and cannot compute official score.")

    for index, item in enumerate(questions, start=1):
        task_id = item.get("task_id", "")
        question = item.get("question", "")
        file_name = item.get("file_name") or None
        print(f"[{index}/{len(questions)}] {task_id} file={file_name or '-'}")

        started_at = time.monotonic()
        error = ""
        try:
            answer = agent.run(question=question, task_id=task_id, file_name=file_name)
        except Exception as exc:
            answer = ""
            error = repr(exc)
        elapsed = time.monotonic() - started_at

        row = {
            "task_id": task_id,
            "file_name": file_name or "",
            "category": categorize_task(question, file_name),
            "question": question,
            "submitted_answer": answer,
            "error": error,
            "local_status": local_status(answer, error),
            "elapsed_seconds": round(elapsed, 2),
        }
        rows.append(row)
        print(f"  answer: {answer[:200]!r}")
        if error:
            print(f"  error: {error}")
        print(f"  elapsed: {elapsed:.1f}s")

        with jsonl_path.open("a", encoding="utf-8") as jsonl_file:
            jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "task_id",
                "file_name",
                "category",
                "question",
                "submitted_answer",
                "error",
                "local_status",
                "elapsed_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    write_report(report_path, rows)
    return jsonl_path, csv_path, report_path


def categorize_task(question: str, file_name: str | None) -> str:
    lower_question = question.lower()
    suffix = Path(file_name or "").suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        return "image_attachment"
    if suffix in {".mp3", ".wav", ".m4a"}:
        return "audio_attachment"
    if suffix in {".xlsx", ".xls", ".csv", ".tsv"}:
        return "spreadsheet_attachment"
    if suffix == ".py":
        return "python_attachment"
    if "youtube.com" in lower_question or "youtu.be" in lower_question:
        return "video_url"
    if any(
        signal in lower_question
        for signal in (
            "wikipedia",
            "article",
            "paper",
            "as of",
            "season",
            "olympics",
            "libretext",
            "universe today",
        )
    ):
        return "web_research"
    return "reasoning"


def local_status(answer: str, error: str) -> str:
    if error:
        return "error"
    if not answer.strip():
        return "blank"
    if answer.lower().startswith("tool error:"):
        return "tool_error"
    if "backend is not configured" in answer.lower():
        return "missing_backend"
    return "answered_unverified"


def write_report(path: Path, rows: list[dict]) -> None:
    total = len(rows)
    status_counts = count_by(rows, "local_status")
    category_counts = count_by(rows, "category")
    lines = [
        "# Local GAIA Dry Run Report",
        "",
        "This is not an official score. GAIA ground truth is hidden, so local runs can only verify execution and inspect answers.",
        "",
        f"Total tasks: {total}",
        "",
        "## Status Counts",
        "",
        *[f"- {key}: {value}" for key, value in sorted(status_counts.items())],
        "",
        "## Category Counts",
        "",
        *[f"- {key}: {value}" for key, value in sorted(category_counts.items())],
        "",
        "## Answers",
        "",
    ]
    for row in rows:
        answer = row["submitted_answer"] or "(blank)"
        error = f"\nError: `{row['error']}`" if row["error"] else ""
        lines.extend(
            [
                f"### {row['task_id']}",
                "",
                f"- Category: `{row['category']}`",
                f"- Status: `{row['local_status']}`",
                f"- File: `{row['file_name'] or '-'}`",
                f"- Elapsed: `{row['elapsed_seconds']}s`",
                f"- Answer: `{answer}`",
                error,
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def count_by(rows: list[dict], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = row.get(key, "")
        counts[value] = counts.get(value, 0) + 1
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local GAIA dry evaluation.")
    parser.add_argument("--api-url", default=os.getenv("GAIA_API_URL", DEFAULT_API_URL))
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--task-id",
        action="append",
        default=None,
        help="Run only this task_id. Can be passed multiple times.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jsonl_path, csv_path, report_path = run_local_eval(
        api_url=args.api_url,
        output_dir=Path(args.output_dir),
        limit=args.limit,
        task_ids=set(args.task_id) if args.task_id else None,
    )
    print(f"Wrote JSONL: {jsonl_path}")
    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
