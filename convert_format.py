"""Convert raw_output.jsonl to LLaVA conversation format.

Per raw record emits:
  - 1 caption entry
  - 3 VQA entries
  - 1 activity entry (if activity is not null)

Output: data/llava_dataset.jsonl  (shuffled)
"""

import json
import random
from pathlib import Path

RAW_OUTPUT_PATH = Path("data/raw_output.jsonl")
LLAVA_OUTPUT_PATH = Path("data/llava_dataset.jsonl")

CAPTION_QUESTION = "Hãy mô tả hình ảnh này bằng tiếng Việt."
ACTIVITY_QUESTION = "Hoạt động nào đang diễn ra trong hình ảnh này?"


def make_caption_entry(record: dict) -> dict:
    return {
        "image": record["image"],
        "conversations": [
            {"from": "human", "value": CAPTION_QUESTION},
            {"from": "gpt", "value": record["caption"]},
        ],
    }


def make_vqa_entries(record: dict) -> list[dict]:
    entries = []
    for pair in record.get("vqa", []):
        entries.append({
            "image": record["image"],
            "conversations": [
                {"from": "human", "value": pair["question"]},
                {"from": "gpt", "value": pair["answer"]},
            ],
        })
    return entries


def make_activity_entry(record: dict) -> dict | None:
    activity = record.get("activity")
    if not activity:
        return None
    return {
        "image": record["image"],
        "conversations": [
            {"from": "human", "value": ACTIVITY_QUESTION},
            {"from": "gpt", "value": activity},
        ],
    }


def convert() -> None:
    if not RAW_OUTPUT_PATH.exists():
        print(f"[error] {RAW_OUTPUT_PATH} not found. Run generate_dataset.py first.")
        return

    all_entries: list[dict] = []

    with open(RAW_OUTPUT_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[warn] Skipping malformed line: {exc}")
                continue

            all_entries.append(make_caption_entry(record))
            all_entries.extend(make_vqa_entries(record))
            activity_entry = make_activity_entry(record)
            if activity_entry:
                all_entries.append(activity_entry)

    random.shuffle(all_entries)

    LLAVA_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LLAVA_OUTPUT_PATH, "w") as f:
        for entry in all_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"[done] {len(all_entries)} LLaVA entries → {LLAVA_OUTPUT_PATH}")


if __name__ == "__main__":
    convert()
