"""Generate Vietnamese captions + VQA pairs for all frames in the manifest.

Usage:
    python generate_dataset.py [--limit N] [--resume]

Outputs:
    data/raw_output.jsonl     — successful records
    data/failed.jsonl         — failures with reason + raw_response
    data/checkpoint.json      — set of processed frame_paths
"""

import argparse
import asyncio
import base64
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm

from prompts import build_prompt

load_dotenv()

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))
MANIFEST_PATH = Path("data/frames_manifest.jsonl")
RAW_OUTPUT_PATH = Path("data/raw_output.jsonl")
FAILED_PATH = Path("data/failed.jsonl")
CHECKPOINT_PATH = Path("data/checkpoint.json")

GENERATION_MODEL = os.getenv("GENERATION_MODEL", "google/gemini-2.0-flash-001")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

PRICES: dict[str, tuple[float, float]] = {
    "google/gemini-2.0-flash-001": (0.075 / 1e6, 0.30 / 1e6),
    "qwen/qwen2.5-vl-32b-instruct": (0.40 / 1e6, 0.40 / 1e6),
}

MAX_CONCURRENT = 50
MAX_RETRIES = 5
CHECKPOINT_EVERY = 1000


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

def extract_json(text: str) -> dict:
    """Strip markdown fences and parse JSON from model response."""
    text = re.sub(r"^```json\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    data = json.loads(text)
    assert "caption" in data, "Missing 'caption' key"
    assert "vqa" in data, "Missing 'vqa' key"
    assert len(data["vqa"]) == 3, f"Expected 3 VQA pairs, got {len(data['vqa'])}"
    return data


# ---------------------------------------------------------------------------
# API call with exponential backoff
# ---------------------------------------------------------------------------

async def call_with_backoff(
    client: AsyncOpenAI,
    model_id: str,
    image_b64: str,
    prompt: str,
) -> tuple[str, int, int]:
    """Call the model with retry on 429. Returns (text, in_tokens, out_tokens)."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = await client.chat.completions.create(
                model=model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_b64}},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                max_tokens=1024,
            )
            text = resp.choices[0].message.content or ""
            in_tok = resp.usage.prompt_tokens if resp.usage else 0
            out_tok = resp.usage.completion_tokens if resp.usage else 0
            return text, in_tok, out_tok
        except Exception as exc:
            err_str = str(exc)
            if "429" in err_str and attempt < MAX_RETRIES - 1:
                sleep_time = 2 ** attempt
                await asyncio.sleep(sleep_time)
            else:
                raise
    raise RuntimeError("Max retries exceeded")


# ---------------------------------------------------------------------------
# Main worker
# ---------------------------------------------------------------------------

async def process_frame(
    semaphore: asyncio.Semaphore,
    client: AsyncOpenAI,
    rec: dict,
    model_id: str,
    raw_file,
    failed_file,
    counters: dict,
) -> None:
    frame_path = rec["frame_path"]
    full_path = OUTPUT_DIR / frame_path

    if not full_path.exists():
        async with semaphore:
            failed_file.write(json.dumps({
                "image": frame_path,
                "reason": "file_not_found",
                "raw_response": "",
            }, ensure_ascii=False) + "\n")
            counters["failed"] += 1
        return

    with open(full_path, "rb") as f:
        data = f.read()
    ext = full_path.suffix.lstrip(".")
    mime = f"image/{'jpeg' if ext in ('jpg', 'jpeg') else ext}"
    image_b64 = f"data:{mime};base64,{base64.b64encode(data).decode()}"

    prompt = build_prompt(rec.get("has_person", False))

    async with semaphore:
        try:
            text, in_tok, out_tok = await call_with_backoff(client, model_id, image_b64, prompt)
            parsed = extract_json(text)
            in_price, out_price = PRICES.get(model_id, (0.0, 0.0))
            cost = in_tok * in_price + out_tok * out_price
            counters["cost"] += cost

            record = {
                "image": frame_path,
                "source": rec.get("source", ""),
                "has_person": rec.get("has_person", False),
                "caption": parsed["caption"],
                "vqa": parsed["vqa"],
                "activity": parsed.get("activity"),
                "model": model_id,
                "tokens": {"input": in_tok, "output": out_tok},
            }
            raw_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            counters["processed"] += 1

        except Exception as exc:
            raw_resp = locals().get("text", "")
            failed_file.write(json.dumps({
                "image": frame_path,
                "reason": str(exc)[:300],
                "raw_response": raw_resp[:500] if isinstance(raw_resp, str) else "",
            }, ensure_ascii=False) + "\n")
            counters["failed"] += 1

        counters["checkpoint_buf"].append(frame_path)


async def run_generation(limit: int | None = None) -> None:
    # Load manifest
    if not MANIFEST_PATH.exists():
        print(f"[error] Manifest not found: {MANIFEST_PATH}")
        return

    with open(MANIFEST_PATH) as f:
        records = [json.loads(line) for line in f if line.strip()]

    # Load checkpoint
    processed_set: set[str] = set()
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            cp = json.load(f)
            processed_set = set(cp.get("processed", []))
        print(f"[checkpoint] Resuming — {len(processed_set)} already processed")

    # Filter out already processed
    remaining = [r for r in records if r["frame_path"] not in processed_set]
    if limit is not None:
        remaining = remaining[:limit]

    print(f"[generate] {len(remaining)} frames to process with {GENERATION_MODEL}")

    Path("data").mkdir(parents=True, exist_ok=True)

    client = AsyncOpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )

    counters = {
        "processed": 0,
        "failed": 0,
        "cost": 0.0,
        "checkpoint_buf": [],
    }

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    with (
        open(RAW_OUTPUT_PATH, "a") as raw_f,
        open(FAILED_PATH, "a") as failed_f,
        tqdm(total=len(remaining), desc="Generating") as pbar,
    ):
        tasks = []
        for rec in remaining:
            task = asyncio.create_task(
                process_frame(semaphore, client, rec, GENERATION_MODEL, raw_f, failed_f, counters)
            )
            tasks.append((rec["frame_path"], task))

        checkpoint_counter = 0
        for frame_path, task in tasks:
            await task
            pbar.update(1)
            checkpoint_counter += 1

            # Update tqdm postfix
            pbar.set_postfix_str(
                f"processed={counters['processed']} | "
                f"failed={counters['failed']} | "
                f"${counters['cost']:.4f}"
            )

            # Flush checkpoint every N
            if checkpoint_counter % CHECKPOINT_EVERY == 0:
                processed_set.update(counters["checkpoint_buf"])
                counters["checkpoint_buf"].clear()
                with open(CHECKPOINT_PATH, "w") as cp_f:
                    json.dump({"processed": list(processed_set)}, cp_f)
                raw_f.flush()
                failed_f.flush()

    # Final checkpoint
    processed_set.update(counters["checkpoint_buf"])
    with open(CHECKPOINT_PATH, "w") as cp_f:
        json.dump({"processed": list(processed_set)}, cp_f)

    print(f"\n[done] processed={counters['processed']} failed={counters['failed']} cost=${counters['cost']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Vietnamese captions and VQA pairs")
    parser.add_argument("--limit", type=int, default=None, help="Max frames to process")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint (default behaviour)")
    args = parser.parse_args()

    asyncio.run(run_generation(limit=args.limit))


if __name__ == "__main__":
    main()
