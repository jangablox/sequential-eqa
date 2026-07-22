#!/usr/bin/env python3
"""Grade canonical Sequential-EQA predictions with a local Qwen model."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Callable, Iterable

RUBRIC = """You are grading an embodied question-answering response.
Return one integer from 1 to 5 and no other text.
5: the response perfectly matches the reference answer.
1: the response is completely different from the reference answer.

Question: {question}
Reference answer: {reference}
Response: {prediction}
Score:"""


def parse_score(text: str) -> int:
    match = re.fullmatch(r"\s*([1-5])\s*", text)
    if not match:
        raise ValueError(f"Grader did not return a single score from 1 to 5: {text!r}")
    return int(match.group(1))


class QwenGrader:
    def __init__(self, model_id: str) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype="auto", device_map="auto"
        )

    def __call__(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        try:
            text = self.tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except TypeError:
            text = self.tokenizer.apply_chat_template(messages, **kwargs)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated = self.model.generate(**inputs, max_new_tokens=8, do_sample=False)
        output = generated[0][inputs.input_ids.shape[1] :]
        return self.tokenizer.decode(output, skip_special_tokens=True).strip()


def grade_records(
    records: Iterable[dict[str, object]], grader: Callable[[str], str], retries: int = 1
) -> list[dict[str, object]]:
    graded: list[dict[str, object]] = []
    for original in records:
        row = dict(original)
        if row.get("score") is not None:
            graded.append(row)
            continue
        prompt = RUBRIC.format(
            question=str(row.get("question", "")),
            reference=str(row.get("reference_answer", "")),
            prediction=str(row.get("prediction", ""))[:1500],
        )
        error: Exception | None = None
        for _ in range(retries + 1):
            try:
                row["score"] = parse_score(grader(prompt))
                row.pop("grading_error", None)
                error = None
                break
            except Exception as exc:  # Preserve the row and a useful failure reason.
                error = exc
        if error is not None:
            row["score"] = None
            row["grading_error"] = str(error)
        graded.append(row)
    return graded


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3-8B-FP8")
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--on-error", choices=("fail", "keep"), default="fail")
    args = parser.parse_args()

    source = read_jsonl(args.input)
    if args.resume and args.output.exists():
        completed = {row["question_id"]: row for row in read_jsonl(args.output) if row.get("score") is not None}
        source = [completed.get(str(row.get("question_id")), row) for row in source]
    rows = grade_records(source, QwenGrader(args.model), retries=args.retries)
    write_jsonl(args.output, rows)
    errors = sum(row.get("score") is None for row in rows)
    if errors and args.on_error == "fail":
        raise SystemExit(f"Grading completed with {errors} errors; details are saved in {args.output}")
    print(f"Graded {len(rows) - errors}/{len(rows)} records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

