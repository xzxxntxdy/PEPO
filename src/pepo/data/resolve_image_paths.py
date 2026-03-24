"""Resolve relative image paths inside JSON or JSONL annotations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def resolve_path(path: str, image_root: str) -> str:
    if not isinstance(path, str) or not path:
        return path
    if os.path.isabs(path):
        return path
    return str(Path(image_root) / path)


def rewrite_record(record: Any, image_root: str) -> Any:
    if not isinstance(record, dict):
        return record
    if "images" in record and isinstance(record["images"], list):
        record["images"] = [resolve_path(path, image_root) for path in record["images"]]
    if "image" in record and isinstance(record["image"], str):
        record["image"] = resolve_path(record["image"], image_root)
    return record


def process_jsonl(input_path: Path, output_path: Path, image_root: str) -> None:
    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = rewrite_record(json.loads(line), image_root)
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")


def process_json(input_path: Path, output_path: Path, image_root: str) -> None:
    with input_path.open("r", encoding="utf-8") as fin:
        payload = json.load(fin)
    if isinstance(payload, list):
        payload = [rewrite_record(item, image_root) for item in payload]
    else:
        payload = rewrite_record(payload, image_root)
    with output_path.open("w", encoding="utf-8") as fout:
        json.dump(payload, fout, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve relative image paths to absolute paths.")
    parser.add_argument("--input", required=True, help="Input JSON or JSONL annotation file.")
    parser.add_argument("--output", required=True, help="Output JSON or JSONL annotation file.")
    parser.add_argument("--image_root", required=True, help="Root directory that contains the raw images.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if input_path.suffix == ".jsonl":
        process_jsonl(input_path, output_path, args.image_root)
    elif input_path.suffix == ".json":
        process_json(input_path, output_path, args.image_root)
    else:
        raise ValueError(f"Unsupported file type: {input_path}")


if __name__ == "__main__":
    main()

