from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


SAFETY_KEYWORDS = {
    "Violence": [
        "weapon",
        "kill",
        "attack",
        "bomb",
        "fight",
        "assault",
        "violence",
        "harm",
        "injury",
        "shoot",
    ],
    "Drugs": [
        "drug",
        "cocaine",
        "heroin",
        "meth",
        "opioid",
        "narcotic",
        "overdose",
        "illegal substance",
    ],
    "Self-harm": [
        "suicide",
        "self-harm",
        "self harm",
        "cutting",
        "overdose",
        "hurt myself",
        "end my life",
    ],
    "Privacy": [
        "password",
        "credential",
        "private",
        "personal data",
        "dox",
        "address",
        "ssn",
        "social security",
        "credit card",
    ],
    "Fraud": [
        "fraud",
        "scam",
        "phishing",
        "steal",
        "fake id",
        "counterfeit",
        "forgery",
        "hack",
        "malware",
    ],
    "Hate": [
        "hate",
        "racist",
        "sexist",
        "slur",
        "holocaust",
        "genocide",
        "discriminate",
        "inferior",
    ],
}

TASK_KEYWORDS = {
    "Coding": ["code", "python", "javascript", "program", "function", "algorithm", "debug", "sql"],
    "Math": ["calculate", "equation", "probability", "number", "solve", "geometry", "algebra"],
    "Reasoning": ["reason", "logic", "infer", "deduce", "explain why", "step by step"],
    "Writing": ["write", "summarize", "rewrite", "story", "email", "essay", "poem", "blog"],
    "QA": ["what", "who", "when", "where", "why", "how", "answer", "question"],
}


def read_records(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("train", "data", "examples"):
                if isinstance(data.get(key), list):
                    return data[key]
        raise ValueError(f"Unsupported JSON shape in {path}")
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"Unsupported input extension: {path.suffix}")


def record_text(record: Dict[str, Any], text_fields: Iterable[str] | None = None) -> str:
    if text_fields:
        values = [str(record.get(field, "")) for field in text_fields]
    elif "messages" in record and isinstance(record["messages"], list):
        values = [str(message.get("content", "")) for message in record["messages"]]
    else:
        values = [
            str(record.get(field, ""))
            for field in ("instruction", "input", "output", "prompt", "completion", "chosen", "rejected", "text")
        ]
    return "\n".join(value for value in values if value).strip()


def infer_label(text: str, keyword_map: Dict[str, List[str]], default: str) -> str:
    lowered = text.lower()
    scores = {
        label: sum(1 for keyword in keywords if keyword in lowered)
        for label, keywords in keyword_map.items()
    }
    best_label, best_score = max(scores.items(), key=lambda item: item[1])
    return best_label if best_score > 0 else default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--kind", choices=["reference", "task"], default="reference")
    parser.add_argument("--text-fields", nargs="*", default=None)
    parser.add_argument("--category-field", default=None, help="Use an existing category/group field when present.")
    parser.add_argument("--risk-field", default=None, help="Copy an existing numeric risk score field when present.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_records(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    keyword_map = SAFETY_KEYWORDS if args.kind == "reference" else TASK_KEYWORDS
    label_key = "safety_category" if args.kind == "reference" else "task_group"
    default_label = "Other" if args.kind == "reference" else "QA"

    with args.output.open("w", encoding="utf-8") as f:
        for idx, record in enumerate(records):
            text = record_text(record, args.text_fields)
            existing_label = record.get(args.category_field) if args.category_field else None
            row: Dict[str, Any] = {
                "index": idx,
                label_key: existing_label or infer_label(text, keyword_map, default_label),
            }
            if args.risk_field and args.risk_field in record:
                try:
                    row["risk_score"] = float(record[args.risk_field])
                except (TypeError, ValueError):
                    pass
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} metadata rows to {args.output}")


if __name__ == "__main__":
    main()

