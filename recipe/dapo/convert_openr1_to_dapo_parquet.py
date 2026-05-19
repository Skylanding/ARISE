#!/usr/bin/env python3
import argparse
import json
import os
import random
from typing import Any, Dict, List, Optional

import pandas as pd
from datasets import load_dataset


def _extract_problem_text(row: Dict[str, Any]) -> str:
    problem = row.get("problem")
    if isinstance(problem, str) and problem.strip():
        return problem.strip()

    messages = row.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            if str(msg.get("role", "")).lower() != "user":
                continue
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()

    return ""


def _extract_answer_text(row: Dict[str, Any]) -> str:
    answer = row.get("answer")
    if isinstance(answer, str):
        return answer.strip()
    if answer is None:
        return ""
    return str(answer).strip()


def _build_prompt(problem_text: str) -> List[Dict[str, str]]:
    return [{"role": "user", "content": problem_text}]


def convert(
    repo_id: str,
    subset: str,
    split: str,
    output_path: str,
    sample_ratio: float,
    max_samples: Optional[int],
    seed: int,
) -> None:
    ds = load_dataset(repo_id, subset, split=split)
    total = len(ds)
    indices = list(range(total))

    rng = random.Random(seed)
    rng.shuffle(indices)

    if sample_ratio < 1.0:
        keep = max(1, int(total * sample_ratio))
        indices = indices[:keep]

    if max_samples is not None:
        indices = indices[: max(0, max_samples)]

    rows_out: List[Dict[str, Any]] = []
    dropped = 0

    for new_idx, src_idx in enumerate(indices):
        row = ds[int(src_idx)]
        problem_text = _extract_problem_text(row)
        answer_text = _extract_answer_text(row)

        if not problem_text or not answer_text:
            dropped += 1
            continue

        extra_info = {
            "index": int(new_idx),
            "source_dataset": repo_id,
            "subset": subset,
            "split": split,
            "problem_type": row.get("problem_type"),
            "question_type": row.get("question_type"),
            "source": row.get("source"),
            "uuid": row.get("uuid"),
        }

        rows_out.append(
            {
                "prompt": _build_prompt(problem_text),
                "ground_truth": answer_text,
                "answer": answer_text,
                "data_source": "math_dapo",
                "extra_info": extra_info,
            }
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(rows_out)
    df.to_parquet(output_path, index=False)

    print(
        json.dumps(
            {
                "repo_id": repo_id,
                "subset": subset,
                "split": split,
                "output_path": output_path,
                "input_rows": total,
                "selected_rows_before_filter": len(indices),
                "written_rows": len(rows_out),
                "dropped_rows": dropped,
            },
            ensure_ascii=True,
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert OpenR1 math dataset to DAPO-compatible parquet.")
    parser.add_argument("--repo-id", default="open-r1/OpenR1-Math-220k")
    parser.add_argument("--subset", default="default")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--output",
        default="/home/ubuntu/verl/data/openr1-math-default-1pct-dapo.parquet",
    )
    parser.add_argument("--sample-ratio", type=float, default=0.01)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not (0 < args.sample_ratio <= 1.0):
        raise ValueError("--sample-ratio must be in (0, 1].")

    convert(
        repo_id=args.repo_id,
        subset=args.subset,
        split=args.split,
        output_path=args.output,
        sample_ratio=args.sample_ratio,
        max_samples=args.max_samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
