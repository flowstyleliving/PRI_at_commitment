#!/usr/bin/env python3
"""Generate TriviaQA paired-prompt dataset for Step 3 factual-rung pilot.

For each selected TriviaQA question, produces two prompts:
  label=0  correct answer injected   (model should answer YES)
  label=1  wrong answer cross-sampled (model should answer NO)

Analogous JSONL schema to the ANLI slice used in Steps 1-2 so the
calibrator and belief-readout scripts can consume it unchanged.

Wrong answers are drawn by cross-sampling the canonical answer from a
different randomly-selected question in the same split. Collision guard
checks that the wrong answer does not appear in the correct question's
alias set (case-insensitive).

Usage
-----
# n=20 pilot (inspect before scaling)
python scripts/generate_triviaqa_paired.py \
    --n-questions 10 \
    --output experiments/triviaqa-paired/2026-05-25/pilot_n20.jsonl \
    --no-shuffle

# n=100 main dataset
python scripts/generate_triviaqa_paired.py \
    --n-questions 50 \
    --output experiments/triviaqa-paired/2026-05-25/n100.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

PROMPT_TEMPLATE = (
    "Instruction: Read the question and proposed answer, then decide whether "
    "the proposed answer is correct. Answer YES if the proposed answer is "
    "correct, NO if the proposed answer is incorrect.\n\n"
    "Question: {question}\nProposed answer: {answer}\nAnswer:"
)

SCHEMA_VERSION = "triviaqa_paired_v1"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical(item: dict[str, Any]) -> str:
    return item["answer"]["value"].strip()


def _all_aliases(item: dict[str, Any]) -> set[str]:
    aliases = item["answer"].get("aliases", [])
    value = item["answer"]["value"]
    return {a.lower().strip() for a in aliases} | {value.lower().strip()}


def _load_dataset(split: str) -> Any:
    from datasets import load_dataset  # type: ignore
    return load_dataset("trivia_qa", "rc.wikipedia", split=split)


def _select_pairs(
    pool: list[dict],
    n_questions: int,
    rng: random.Random,
) -> list[dict]:
    """Return up to n_questions dicts each with correct + wrong answer."""
    pairs: list[dict] = []
    used_qids: set[str] = set()
    # Build a shuffled donor pool for each candidate so wrong answers are diverse
    donor_pool = list(pool)

    for item in pool:
        if len(pairs) >= n_questions:
            break
        qid = item["question_id"]
        if qid in used_qids:
            continue

        correct = _canonical(item)
        correct_set = _all_aliases(item)

        # Cross-sample wrong answer from a randomly-ordered donor pool
        rng.shuffle(donor_pool)
        wrong: str | None = None
        for other in donor_pool:
            if other["question_id"] == qid:
                continue
            candidate = _canonical(other)
            if candidate.lower().strip() in correct_set:
                continue  # accidentally correct — skip
            if len(candidate.strip()) < 2:
                continue  # degenerate
            # Avoid re-using an answer already picked as wrong for another question
            already_used = any(p["wrong_answer"] == candidate for p in pairs)
            if already_used:
                continue
            wrong = candidate
            break

        if wrong is None:
            continue  # couldn't find a clean unique wrong answer — skip

        used_qids.add(qid)
        pairs.append({
            "question": item["question"].strip(),
            "correct_answer": correct,
            "wrong_answer": wrong,
            "question_id": qid,
        })

    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-questions", type=int, default=50,
                        help="Number of unique questions (dataset size = 2× this)")
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output", required=True)
    parser.add_argument("--no-shuffle", action="store_true",
                        help="Keep correct+wrong pairs adjacent (useful for pilot inspection)")
    parser.add_argument("--pool-multiplier", type=int, default=10,
                        help="Sample pool_multiplier × n_questions items before filtering")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading TriviaQA rc.wikipedia / {args.split}...")
    ds = _load_dataset(args.split)
    print(f"Dataset size: {len(ds)}")

    pool_size = min(len(ds), args.n_questions * args.pool_multiplier)
    indices = rng.sample(range(len(ds)), pool_size)
    pool = [ds[i] for i in indices]

    pairs = _select_pairs(pool, args.n_questions, rng)
    print(f"Selected {len(pairs)} pairs (target {args.n_questions})")

    records: list[dict] = []
    for pair in pairs:
        # correct answer — label 0 (YES expected; analogous to entailment in ANLI)
        records.append({
            "prompt": PROMPT_TEMPLATE.format(
                question=pair["question"],
                answer=pair["correct_answer"],
            ),
            "label": 0,
            "meta": {
                "question_id": pair["question_id"],
                "correct_answer": pair["correct_answer"],
                "wrong_answer": pair["wrong_answer"],
                "kind": "correct",
            },
        })
        # wrong answer — label 1 (NO expected; analogous to contradiction in ANLI)
        records.append({
            "prompt": PROMPT_TEMPLATE.format(
                question=pair["question"],
                answer=pair["wrong_answer"],
            ),
            "label": 1,
            "meta": {
                "question_id": pair["question_id"],
                "correct_answer": pair["correct_answer"],
                "wrong_answer": pair["wrong_answer"],
                "kind": "wrong",
            },
        })

    if not args.no_shuffle:
        rng.shuffle(records)

    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    data_hash = _sha256_file(out_path)
    n_label1 = sum(r["label"] for r in records)
    n_label0 = len(records) - n_label1

    print(f"Wrote {len(records)} samples → {out_path}")
    print(f"  label=0 (correct / YES expected): {n_label0}")
    print(f"  label=1 (wrong / NO expected):    {n_label1}")
    print(f"  data_hash_sha256: {data_hash}")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "n_questions": len(pairs),
        "n_samples": len(records),
        "seed": args.seed,
        "split": args.split,
        "hf_dataset": "trivia_qa",
        "hf_config": "rc.wikipedia",
        "data_hash_sha256": data_hash,
        "output": str(out_path.resolve()),
        "shuffled": not args.no_shuffle,
        "prompt_template": PROMPT_TEMPLATE,
        "label_convention": {
            "0": "correct_answer — model should answer YES",
            "1": "wrong_answer — model should answer NO (contradiction analog)",
        },
    }
    manifest_path = out_path.with_suffix(".manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  manifest: {manifest_path}")


if __name__ == "__main__":
    main()
