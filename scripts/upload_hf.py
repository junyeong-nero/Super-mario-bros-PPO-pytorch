"""
Collect logs for a given world and stage, build a Hugging Face Dataset, and upload it.

Usage:
    python scripts/upload_hf.py --world 1 --stage 1 --repo_id yourname/mario-logs
Optional:
    --log_dir logs --hf_token <token or rely on HF_TOKEN env> --push: disable with --no-push
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import Dataset


def load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}


def find_log_dir(base: Path, world: int, stage: int) -> Path:
    """Finds the log directory, supporting both hyphen and underscore separators."""
    candidates = [
        base / f"{world}-{stage}",
        base / f"{world}_{stage}",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"No log directory found for world={world}, stage={stage} under {base}"
    )


def collect_records(log_dir: Path) -> List[Dict[str, Any]]:
    """Flattens all log JSON files into a list of dataset rows."""
    records: List[Dict[str, Any]] = []
    json_files = sorted(log_dir.glob("*.json"))

    if not json_files:
        raise FileNotFoundError(f"No JSON logs found in {log_dir}")

    for path in json_files:
        log_data = load_json(path)
        log_index = path.stem
        history: List[Dict[str, Any]] = log_data.get("history", [])

        for step_idx, entry in enumerate(history):
            record = {
                "world": log_data.get("world"),
                "stage": log_data.get("stage"),
                "action_type": log_data.get("action_type"),
                "action_level": entry.get("reward", {}).get("action", 0),
                "log_index": log_index,
                "step": step_idx,
                "time": entry.get("time"),
                "info": entry.get("info"),
                "reward": entry.get("reward"),
                "observation_text": entry.get("observation"),
                "objects": entry.get("objects"),
            }
            records.append(record)

    return records


def deduplicate_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Removes rows with identical observation dictionaries while preserving order."""
    deduped: List[Dict[str, Any]] = []
    seen_keys = set()

    for record in records:
        objects_key = json.dumps(
            record.get("objects"),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )

        if objects_key in seen_keys:
            continue

        seen_keys.add(objects_key)
        deduped.append(record)

    return deduped


def build_dataset(records: List[Dict[str, Any]]) -> Dataset:
    return Dataset.from_list(records)


def push_dataset(dataset: Dataset, repo_id: str, token: Optional[str], private: bool):
    dataset.push_to_hub(repo_id=repo_id, token=token, private=private)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload Mario PPO logs as a Hugging Face Dataset"
    )
    parser.add_argument("--world", type=int, required=True, help="World number")
    parser.add_argument("--stage", type=int, required=True, help="Stage number")
    parser.add_argument(
        "--log_dir", type=str, default="logs", help="Base directory containing logs"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Target Hugging Face dataset repo id (e.g., user/mario-logs)",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Upload dataset as private (default: public)",
    )
    parser.add_argument(
        "--push",
        dest="push",
        action="store_true",
        default=True,
        help="Push to hub (disable with --no-push)",
    )
    parser.add_argument(
        "--no-push",
        dest="push",
        action="store_false",
        help="Do not push, just build locally",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    base_log_dir = Path(args.log_dir)
    log_dir = find_log_dir(base_log_dir, args.world, args.stage)
    print(f"Using log directory: {log_dir}")

    records = collect_records(log_dir)
    total_records = len(records)
    print(
        f"Collected {total_records} records from {len(list(log_dir.glob('*.json')))} files."
    )

    records = deduplicate_records(records)
    deduped_records = len(records)
    if deduped_records != total_records:
        print(
            f"Deduplicated records by observation dictionary: {deduped_records} remaining."
        )

    dataset = build_dataset(records)

    if args.push:
        token = args.hf_token or os.environ.get("HF_TOKEN")
        if not token:
            raise ValueError(
                "Hugging Face token not provided. Use --hf_token or set HF_TOKEN."
            )
        print(f"Pushing dataset to hub: {args.repo_id} (private={args.private})")
        print(dataset)
        push_dataset(dataset, args.repo_id, token, args.private)
        print("Upload complete.")
    else:
        print("Push disabled; dataset built locally.")


if __name__ == "__main__":
    main()
