"""Sample 20 frames from manifest and compare two VLM models side-by-side.

Runs both models concurrently per image, prints a comparison table,
shows cost summary + 100K projection, then prompts user to pick a model
and writes GENERATION_MODEL=<id> to .env.
"""

import asyncio
import base64
import json
import os
import random
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from prompts import build_prompt

load_dotenv()

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))
MANIFEST_PATH = Path("data/frames_manifest.jsonl")
ENV_PATH = Path(".env")

MODELS = {
    "gemini": "google/gemini-2.0-flash-001",
    "qwen": "qwen/qwen2.5-vl-32b-instruct",
}

PRICES: dict[str, tuple[float, float]] = {
    "google/gemini-2.0-flash-001": (0.075 / 1e6, 0.30 / 1e6),
    "qwen/qwen2.5-vl-32b-instruct": (0.40 / 1e6, 0.40 / 1e6),
}

SAMPLE_SIZE = 20


def load_manifest() -> list[dict]:
    if not MANIFEST_PATH.exists():
        print(f"[warn] manifest not found: {MANIFEST_PATH}")
        return []
    with open(MANIFEST_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def stratified_sample(records: list[dict], n: int = SAMPLE_SIZE) -> list[dict]:
    by_source: dict[str, list[dict]] = {}
    for rec in records:
        by_source.setdefault(rec["source"], []).append(rec)

    sources = list(by_source.keys())
    per_source = n // len(sources) if sources else n
    remainder = n - per_source * len(sources)

    sampled: list[dict] = []
    for i, src in enumerate(sources):
        k = per_source + (1 if i < remainder else 0)
        sampled.extend(random.sample(by_source[src], min(k, len(by_source[src]))))
    return sampled[:n]


def image_to_base64(frame_path: str) -> str | None:
    full_path = OUTPUT_DIR / frame_path
    if not full_path.exists():
        return None
    with open(full_path, "rb") as f:
        data = f.read()
    ext = full_path.suffix.lstrip(".")
    mime = f"image/{'jpeg' if ext in ('jpg', 'jpeg') else ext}"
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


async def call_model(
    client: AsyncOpenAI,
    model_id: str,
    image_b64: str,
    prompt: str,
) -> tuple[str, int, int, float]:
    """Call model, return (text, input_tokens, output_tokens, latency_s)."""
    t0 = time.monotonic()
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
    latency = time.monotonic() - t0
    text = resp.choices[0].message.content or ""
    in_tok = resp.usage.prompt_tokens if resp.usage else 0
    out_tok = resp.usage.completion_tokens if resp.usage else 0
    return text, in_tok, out_tok, latency


def compute_cost(model_id: str, in_tok: int, out_tok: int) -> float:
    in_price, out_price = PRICES.get(model_id, (0.0, 0.0))
    return in_tok * in_price + out_tok * out_price


def print_comparison_table(
    idx: int,
    frame_path: str,
    gemini_text: str,
    qwen_text: str,
    gemini_latency: float,
    qwen_latency: float,
) -> None:
    sep = "-" * 100
    print(f"\n{sep}")
    print(f"Image {idx+1:02d}: {frame_path}")
    print(sep)

    def truncate(s: str, n: int = 300) -> str:
        return s[:n] + "..." if len(s) > n else s

    print(f"  [gemini  {gemini_latency:.1f}s] {truncate(gemini_text)}")
    print(f"  [qwen    {qwen_latency:.1f}s] {truncate(qwen_text)}")


async def run_sample() -> None:
    records = load_manifest()
    if not records:
        print("[error] No records in manifest. Run extract_frames.py first.")
        return

    sampled = stratified_sample(records, SAMPLE_SIZE)
    print(f"Sampled {len(sampled)} frames: " + str({
        src: sum(1 for r in sampled if r["source"] == src)
        for src in set(r["source"] for r in sampled)
    }))

    client = AsyncOpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )

    total_cost: dict[str, float] = {k: 0.0 for k in MODELS}
    total_tokens: dict[str, dict[str, int]] = {k: {"in": 0, "out": 0} for k in MODELS}
    failures: dict[str, int] = {k: 0 for k in MODELS}

    for idx, rec in enumerate(sampled):
        b64 = image_to_base64(rec["frame_path"])
        if b64 is None:
            print(f"[skip] image not found: {rec['frame_path']}")
            continue

        prompt = build_prompt(rec.get("has_person", False))

        gemini_task = call_model(client, MODELS["gemini"], b64, prompt)
        qwen_task = call_model(client, MODELS["qwen"], b64, prompt)

        results = await asyncio.gather(gemini_task, qwen_task, return_exceptions=True)

        gemini_text, qwen_text = "", ""
        gemini_latency, qwen_latency = 0.0, 0.0

        for key, res in zip(("gemini", "qwen"), results):
            if isinstance(res, Exception):
                print(f"  [{key}] ERROR: {res}")
                failures[key] += 1
            else:
                text, in_tok, out_tok, lat = res
                if key == "gemini":
                    gemini_text, gemini_latency = text, lat
                else:
                    qwen_text, qwen_latency = text, lat
                cost = compute_cost(MODELS[key], in_tok, out_tok)
                total_cost[key] += cost
                total_tokens[key]["in"] += in_tok
                total_tokens[key]["out"] += out_tok

        print_comparison_table(idx, rec["frame_path"], gemini_text, qwen_text, gemini_latency, qwen_latency)

    # Cost summary
    n = len(sampled)
    print("\n" + "=" * 80)
    print("COST SUMMARY")
    print("=" * 80)
    print(f"{'Model':<40} {'Sample cost':>12} {'100K projection':>18} {'Failures':>10}")
    print("-" * 80)
    for key, model_id in MODELS.items():
        sample_cost = total_cost[key]
        projected = (sample_cost / n * 100_000) if n > 0 else 0
        print(f"{model_id:<40} ${sample_cost:>10.4f} ${projected:>16.2f} {failures[key]:>10}")
    print("=" * 80)

    # Model selection
    choice = input("\nWhich model? (gemini/qwen): ").strip().lower()
    if choice not in MODELS:
        print(f"[warn] Unknown choice '{choice}', defaulting to gemini")
        choice = "gemini"
    selected_model_id = MODELS[choice]

    # Update .env
    env_content = ENV_PATH.read_text() if ENV_PATH.exists() else ""
    if re.search(r"^GENERATION_MODEL=", env_content, re.MULTILINE):
        env_content = re.sub(
            r"^GENERATION_MODEL=.*$",
            f"GENERATION_MODEL={selected_model_id}",
            env_content,
            flags=re.MULTILINE,
        )
    else:
        env_content += f"\nGENERATION_MODEL={selected_model_id}\n"
    ENV_PATH.write_text(env_content)
    print(f"[.env] GENERATION_MODEL set to: {selected_model_id}")


if __name__ == "__main__":
    asyncio.run(run_sample())
