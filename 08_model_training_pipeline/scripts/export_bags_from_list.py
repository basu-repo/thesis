#!/usr/bin/env python3
"""Export multiple bags listed in a plain text file, one by one."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


THESIS_ROOT = Path.home() / "Documents/Thesis"
PIPELINE_ROOT = THESIS_ROOT / "08_model_training_pipeline"
DEFAULT_LIST = PIPELINE_ROOT / "configs" / "bag_export_list.txt"
DEFAULT_BAGS_ROOT = THESIS_ROOT / "03_dataset" / "bags"
ONE_BAG_SCRIPT = PIPELINE_ROOT / "scripts" / "export_one_bag_to_episode_frames.py"


def load_bag_entries(list_path: Path) -> list[str]:
    entries: list[str] = []
    for raw_line in list_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        entries.append(line)
    return entries


def resolve_bag_path(entry: str, bags_root: Path) -> Path:
    candidate = Path(entry).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (bags_root / entry).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list",
        type=Path,
        default=DEFAULT_LIST,
        help="Text file containing bag folder names or absolute paths.",
    )
    parser.add_argument(
        "--bags-root",
        type=Path,
        default=DEFAULT_BAGS_ROOT,
        help="Base folder used when list entries are bag names only.",
    )
    args = parser.parse_args()

    list_path = args.list.expanduser().resolve()
    bags_root = args.bags_root.expanduser().resolve()

    if not list_path.exists():
        raise RuntimeError(f"Bag list file not found: {list_path}")

    entries = load_bag_entries(list_path)
    if not entries:
        raise RuntimeError(f"No bag entries found in: {list_path}")

    print(f"Bag list: {list_path}")
    print(f"Bag root: {bags_root}")
    print(f"Bag count: {len(entries)}")

    failures = 0
    for index, entry in enumerate(entries, start=1):
        bag_path = resolve_bag_path(entry, bags_root)
        print(f"\n[{index}/{len(entries)}] Exporting {bag_path}")
        result = subprocess.run(
            [
                sys.executable,
                str(ONE_BAG_SCRIPT),
                "--bag",
                str(bag_path),
            ]
        )
        if result.returncode != 0:
            failures += 1
            print(f"Failed: {bag_path}")

    print(f"\nDone. Failures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
