#!/usr/bin/env python3
"""Build one shared sample table and one fixed split file for all 08 models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from datasets.paths import EPISODE_FRAMES_ROOT, SPLITS_ROOT, TRAIN_READY_ROOT  # noqa: E402
from datasets.sample_table import (  # noqa: E402
    build_sample_table,
    load_episode_streams,
    save_or_load_fixed_split,
    save_sample_table,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--past-len", type=int, default=10)
    parser.add_argument("--future-len", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    streams = load_episode_streams(EPISODE_FRAMES_ROOT)
    sample_table = build_sample_table(streams, args.past_len, args.future_len)

    sample_table_path = TRAIN_READY_ROOT / (
        f"sample_table_seed{args.seed}_past{args.past_len}_future{args.future_len}.json"
    )
    split_path = SPLITS_ROOT / (
        f"trajectory_split_seed{args.seed}_past{args.past_len}_future{args.future_len}.json"
    )

    save_sample_table(sample_table, sample_table_path)
    split_info = save_or_load_fixed_split(
        sample_table=sample_table,
        split_path=split_path,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        past_len=args.past_len,
        future_len=args.future_len,
    )

    summary = {
        "stream_count": len(streams),
        "sample_count": len(sample_table),
        "sample_table_path": str(sample_table_path),
        "split_path": str(split_path),
        "split_strategy": split_info["split_strategy"],
        "episode_count": split_info["episode_count"],
        "train_episode_ids": split_info["train_episode_ids"],
        "val_episode_ids": split_info["val_episode_ids"],
        "test_episode_ids": split_info["test_episode_ids"],
        "train_count": len(split_info["train_indices"]),
        "val_count": len(split_info["val_indices"]),
        "test_count": len(split_info["test_indices"]),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
