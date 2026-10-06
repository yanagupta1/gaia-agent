import argparse
import csv
import json
import os
import time
from pathlib import Path

import requests

from agent import GaiaAgent


DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"
DEFAULT_OUTPUT_DIR = "local_runs"


def fetch_questions(api_url: str) -> list[dict]:
    response = requests.get(f"{api_url.rstrip('/')}/questions", timeout=30)
    response.raise_for_status()
    return response.json()


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
) -> tuple[Path, Path]:
    questions = select_questions(fetch_questions(api_url), limit=limit, task_ids=task_ids)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    jsonl_path = output_dir / f"answers-{timestamp}.jsonl"
    csv_path = output_dir / f"answers-{timestamp}.csv"

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
            "question": question,
            "submitted_answer": answer,
            "error": error,
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
                "question",
                "submitted_answer",
                "error",
                "elapsed_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return jsonl_path, csv_path


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
    jsonl_path, csv_path = run_local_eval(
        api_url=args.api_url,
        output_dir=Path(args.output_dir),
        limit=args.limit,
        task_ids=set(args.task_id) if args.task_id else None,
    )
    print(f"Wrote JSONL: {jsonl_path}")
    print(f"Wrote CSV: {csv_path}")


if __name__ == "__main__":
    main()
