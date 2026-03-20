"""Print statistics for the generated dataset.

Reads:
  data/raw_output.jsonl
  data/failed.jsonl
  data/frames_manifest.jsonl
"""

import json
import os
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

RAW_OUTPUT_PATH = Path("data/raw_output.jsonl")
FAILED_PATH = Path("data/failed.jsonl")
MANIFEST_PATH = Path("data/frames_manifest.jsonl")
LLAVA_PATH = Path("data/llava_dataset.jsonl")

PRICES: dict[str, tuple[float, float]] = {
    "google/gemini-2.0-flash-001": (0.075 / 1e6, 0.30 / 1e6),
    "qwen/qwen2.5-vl-32b-instruct": (0.40 / 1e6, 0.40 / 1e6),
}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def sep(char: str = "-", width: int = 70) -> str:
    return char * width


def main() -> None:
    raw = load_jsonl(RAW_OUTPUT_PATH)
    failed = load_jsonl(FAILED_PATH)
    manifest = load_jsonl(MANIFEST_PATH)
    llava = load_jsonl(LLAVA_PATH)

    total_processed = len(raw)
    total_failed = len(failed)
    total_attempted = total_processed + total_failed
    fail_rate = (total_failed / total_attempted * 100) if total_attempted > 0 else 0.0

    print(sep("="))
    print("DATASET STATISTICS")
    print(sep("="))
    print(f"  Manifest frames:     {len(manifest):>10,}")
    print(f"  Processed (success): {total_processed:>10,}")
    print(f"  Failed:              {total_failed:>10,}")
    print(f"  Fail rate:           {fail_rate:>9.1f}%")
    print(f"  LLaVA entries:       {len(llava):>10,}")

    # Per-source breakdown
    source_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"success": 0, "failed": 0})
    for rec in raw:
        source_counts[rec.get("source", "unknown")]["success"] += 1
    for rec in failed:
        # failed records don't have source; join via manifest
        source_counts["unknown"]["failed"] += 1

    # Better: build a frame→source map from manifest
    frame_source: dict[str, str] = {r["frame_path"]: r.get("source", "unknown") for r in manifest}
    source_counts2: dict[str, dict[str, int]] = defaultdict(lambda: {"success": 0, "failed": 0})
    for rec in raw:
        src = frame_source.get(rec.get("image", ""), rec.get("source", "unknown"))
        source_counts2[src]["success"] += 1
    for rec in failed:
        src = frame_source.get(rec.get("image", ""), "unknown")
        source_counts2[src]["failed"] += 1

    print(f"\n{sep()}")
    print("PER-SOURCE BREAKDOWN")
    print(sep())
    print(f"  {'Source':<15} {'Success':>10} {'Failed':>10} {'Total':>10}")
    print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10}")
    for src, counts in sorted(source_counts2.items()):
        s = counts["success"]
        fl = counts["failed"]
        print(f"  {src:<15} {s:>10,} {fl:>10,} {s+fl:>10,}")

    # Caption / Q&A lengths
    caption_lengths = [len(r.get("caption", "")) for r in raw]
    all_qa_lengths = []
    for r in raw:
        for qa in r.get("vqa", []):
            all_qa_lengths.append(len(qa.get("question", "")) + len(qa.get("answer", "")))

    avg_caption = sum(caption_lengths) / len(caption_lengths) if caption_lengths else 0
    avg_qa = sum(all_qa_lengths) / len(all_qa_lengths) if all_qa_lengths else 0

    print(f"\n{sep()}")
    print("TEXT STATISTICS")
    print(sep())
    print(f"  Avg caption length (chars):   {avg_caption:>8.1f}")
    print(f"  Avg Q+A length (chars):       {avg_qa:>8.1f}")

    # Activity population
    with_activity = sum(1 for r in raw if r.get("activity"))
    pct_activity = (with_activity / total_processed * 100) if total_processed > 0 else 0
    print(f"  Records with activity:        {with_activity:>8,} ({pct_activity:.1f}%)")

    # Token cost estimate
    total_cost = 0.0
    model_token_totals: dict[str, dict[str, int]] = defaultdict(lambda: {"in": 0, "out": 0})
    for rec in raw:
        model_id = rec.get("model", "")
        tok = rec.get("tokens", {})
        in_tok = tok.get("input", 0)
        out_tok = tok.get("output", 0)
        model_token_totals[model_id]["in"] += in_tok
        model_token_totals[model_id]["out"] += out_tok
        in_price, out_price = PRICES.get(model_id, (0.0, 0.0))
        total_cost += in_tok * in_price + out_tok * out_price

    print(f"\n{sep()}")
    print("TOKEN COST ESTIMATE")
    print(sep())
    for model_id, totals in model_token_totals.items():
        in_p, out_p = PRICES.get(model_id, (0.0, 0.0))
        cost = totals["in"] * in_p + totals["out"] * out_p
        print(f"  {model_id}")
        print(f"    in={totals['in']:,} out={totals['out']:,} cost=${cost:.4f}")
    print(f"  Total cost: ${total_cost:.4f}")

    # 100K projection
    if total_processed > 0:
        projected = total_cost / total_processed * 100_000
        print(f"  100K projection: ${projected:.2f}")

    print(sep("="))


if __name__ == "__main__":
    main()
