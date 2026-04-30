#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates

"""Convert parquet files to JSONL.

By default, this script converts the following remote parquet files:
1. ${DATA_DIR}/test_math.parquet
2. ${DATA_DIR}/test_codeforces.parquet
3. ${DATA_DIR}/test_humanevalplus.parquet

Example:
    python3 scripts/parquet_to_jsonl.py
    python3 scripts/parquet_to_jsonl.py --overwrite
    python3 scripts/parquet_to_jsonl.py /path/to/custom.parquet --output-dir /tmp/jsonl
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path
from typing import Any


DEFAULT_PARQUET_PATHS = [
    Path("${DATA_DIR}/test_math.parquet"),
    Path(
        "${PROJECT_ROOT}"
        "data_code/huggingface.co/datasets/xuekai/flowrl-data-collection/code_data/test_codeforces.parquet"
    ),
    Path(
        "${PROJECT_ROOT}"
        "data_code/huggingface.co/datasets/xuekai/flowrl-data-collection/code_data/test_humanevalplus.parquet"
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert parquet files to JSONL.")
    parser.add_argument(
        "parquet_paths",
        nargs="*",
        type=Path,
        default=DEFAULT_PARQUET_PATHS,
        help="Parquet files to convert. If omitted, the built-in remote paths are used.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for generated JSONL files. Defaults to each parquet file's parent directory.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Number of rows processed per parquet batch.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output JSONL file.",
    )
    return parser.parse_args()


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def resolve_output_path(parquet_path: Path, output_dir: Path | None) -> Path:
    if output_dir is None:
        return parquet_path.with_suffix(".jsonl")
    return output_dir / f"{parquet_path.stem}.jsonl"


def convert_one_file(parquet_path: Path, output_path: Path, batch_size: int, overwrite: bool) -> int:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("Please install pyarrow first: pip install pyarrow") from exc

    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file does not exist: {parquet_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists, use --overwrite to replace it: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_file = pq.ParquetFile(parquet_path)
    rows_written = 0

    with output_path.open("w", encoding="utf-8") as fout:
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            for record in batch.to_pylist():
                fout.write(json.dumps(to_jsonable(record), ensure_ascii=False))
                fout.write("\n")
                rows_written += 1

    return rows_written


def main() -> None:
    args = parse_args()

    for parquet_path in args.parquet_paths:
        output_path = resolve_output_path(parquet_path, args.output_dir)
        print(f"[START] {parquet_path} -> {output_path}")
        rows_written = convert_one_file(
            parquet_path=parquet_path,
            output_path=output_path,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
        )
        print(f"[DONE] wrote {rows_written} rows to {output_path}")


if __name__ == "__main__":
    main()
