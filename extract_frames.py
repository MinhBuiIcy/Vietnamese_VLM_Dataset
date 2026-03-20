"""Extract frames from ADL and EPIC-Kitchens datasets into OUTPUT_DIR.

ADL:  scene-change + fps=0.5 filter via ffmpeg-python → target 60K frames
EPIC: uniform-sample JPEGs from pre-extracted frames → target 30K frames

Outputs:
    data/frames_manifest.jsonl  — one JSON line per frame
"""

import argparse
import json
import os
import random
import re
from pathlib import Path

import ffmpeg
import scipy.io
from dotenv import load_dotenv
from PIL import Image, ImageDraw
from tqdm import tqdm

load_dotenv()

ADL_DIR = Path(os.getenv("ADL_DIR", ""))
EPIC_DIR = Path(os.getenv("EPIC_DIR", ""))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))
MANIFEST_PATH = Path("data/frames_manifest.jsonl")

ADL_TARGET = 60_000
EPIC_TARGET = 30_000
DRY_RUN_COUNT = 10  # fake frames per source in dry-run mode


def _make_fake_image(out_path: Path, label: str) -> None:
    """Create a small synthetic JPEG for pipeline testing."""
    img = Image.new("RGB", (224, 224), color=(random.randint(50, 200),) * 3)
    draw = ImageDraw.Draw(img)
    draw.text((10, 100), label, fill=(255, 255, 255))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "JPEG")


# ---------------------------------------------------------------------------
# ADL helpers
# ---------------------------------------------------------------------------

def _load_adl_person_frames(ann_path: Path) -> set[int]:
    """Return set of frame indices where a person is present.

    The ADL .mat file stores a struct array; we look for a field that
    contains per-frame labels and treat any non-zero value as person-present.
    """
    try:
        mat = scipy.io.loadmat(str(ann_path), squeeze_me=True, struct_as_record=False)
        # Common field name in ADL annotations
        for key in ("labels", "label", "annotations", "activity_labels"):
            if key in mat:
                arr = mat[key]
                # If it's a struct array, try .activity or .person
                if hasattr(arr, "dtype") and arr.dtype.names:
                    for fname in arr.dtype.names:
                        col = arr[fname]
                        # person-presence column is often named 'activity' or 'object'
                        if "person" in fname.lower() or "activity" in fname.lower():
                            return set(int(i) for i, v in enumerate(col) if v)
                # Plain array: assume non-zero rows → person present
                import numpy as np
                flat = np.array(arr).flatten()
                return set(int(i) for i, v in enumerate(flat) if v)
    except Exception:
        pass
    return set()


def _participant_from_path(path: Path) -> str:
    """Extract participant ID (e.g. 'P01') from path stem."""
    m = re.search(r"(P\d+)", path.stem)
    return m.group(1) if m else "P00"


def extract_adl_frames(dry_run: bool = False) -> list[dict]:
    """Extract frames from all ADL .mp4 files using scene+fps filter."""
    out_dir = OUTPUT_DIR / "adl"

    if dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for i in range(DRY_RUN_COUNT):
            participant = f"P{i+1:02d}"
            fname = f"{participant}_{i:06d}.jpg"
            _make_fake_image(out_dir / fname, f"ADL {participant}")
            records.append({
                "frame_path": f"adl/{fname}",
                "source": "adl",
                "has_person": i % 3 == 0,
            })
        return records

    if not ADL_DIR.exists():
        print(f"[ADL] ADL_DIR not found: {ADL_DIR} — skipping")
        return []

    video_dir = ADL_DIR / "videos"
    ann_dir = ADL_DIR / "ADL_annotations"
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(video_dir.glob("*.mp4"))
    if not videos:
        print(f"[ADL] No .mp4 files found in {video_dir}")
        return []

    records: list[dict] = []

    for video_path in tqdm(videos, desc="ADL videos"):
        participant = _participant_from_path(video_path)
        ann_file = ann_dir / f"{participant}_annotations.mat"
        person_frames = _load_adl_person_frames(ann_file) if ann_file.exists() else set()

        out_pattern = str(out_dir / f"{participant}_%06d.jpg")
        try:
            (
                ffmpeg
                .input(str(video_path))
                .filter("select", "gt(scene\\,0.3)+not(mod(n\\,30))")
                .filter("fps", fps=0.5)
                .output(out_pattern, vsync="vfr", q=2)
                .overwrite_output()
                .run(quiet=True)
            )
        except ffmpeg.Error as exc:
            print(f"[ADL] ffmpeg error on {video_path.name}: {exc.stderr.decode()[:200]}")
            continue

        saved = sorted(out_dir.glob(f"{participant}_*.jpg"))
        for frame_path in saved:
            m = re.search(r"_(\d+)\.jpg$", frame_path.name)
            frame_idx = int(m.group(1)) if m else 0
            records.append({
                "frame_path": f"adl/{frame_path.name}",
                "source": "adl",
                "has_person": frame_idx in person_frames,
            })

    if len(records) > ADL_TARGET:
        records = random.sample(records, ADL_TARGET)

    return records


# ---------------------------------------------------------------------------
# EPIC helpers
# ---------------------------------------------------------------------------

def _load_epic_person_frames(epic_dir: Path) -> set[str]:
    """Return set of relative frame paths where noun == 'person'."""
    person_frames: set[str] = set()
    for csv_path in epic_dir.glob("*.csv"):
        try:
            import csv
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    noun = row.get("noun", row.get("noun_class", "")).lower()
                    if noun == "person":
                        participant = row.get("participant_id", "")
                        video_id = row.get("video_id", "")
                        # EPIC frame paths: frames/P01/P01_01/frame_0000000001.jpg
                        start = int(row.get("start_frame", 0))
                        stop = int(row.get("stop_frame", start))
                        for fi in range(start, stop + 1):
                            rel = f"frames/{participant}/{video_id}/frame_{fi:010d}.jpg"
                            person_frames.add(rel)
        except Exception:
            pass
    return person_frames


def extract_epic_frames(dry_run: bool = False) -> list[dict]:
    """Uniform-sample EPIC-Kitchens pre-extracted JPEG frames."""
    out_dir = OUTPUT_DIR / "epic"

    if dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for i in range(DRY_RUN_COUNT):
            fname = f"P{i+1:02d}_01_frame_{i:010d}.jpg"
            _make_fake_image(out_dir / fname, f"EPIC {i}")
            records.append({
                "frame_path": f"epic/{fname}",
                "source": "epic",
                "has_person": False,
            })
        return records

    if not EPIC_DIR.exists():
        print(f"[EPIC] EPIC_DIR not found: {EPIC_DIR} — skipping")
        return []

    frames_root = EPIC_DIR / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_frames = sorted(frames_root.rglob("*.jpg"))
    if not all_frames:
        print(f"[EPIC] No JPEG frames found under {frames_root}")
        return []

    person_frames = _load_epic_person_frames(EPIC_DIR)

    target = min(EPIC_TARGET, len(all_frames))
    step = max(1, len(all_frames) // target)
    sampled = all_frames[::step][:target]

    records: list[dict] = []
    for fp in tqdm(sampled, desc="EPIC frames"):
        rel = fp.relative_to(EPIC_DIR)
        has_person = str(rel) in person_frames
        dest = out_dir / fp.name
        if not dest.exists():
            try:
                dest.symlink_to(fp.resolve())
            except OSError:
                import shutil
                shutil.copy2(fp, dest)
        records.append({
            "frame_path": f"epic/{fp.name}",
            "source": "epic",
            "has_person": has_person,
        })

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def write_manifest(records: list[dict], append: bool = False) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with open(MANIFEST_PATH, mode) as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[manifest] wrote {len(records)} records → {MANIFEST_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract ADL and EPIC frames")
    parser.add_argument("--dry-run", action="store_true", help="Skip ffmpeg; write fake manifest entries")
    parser.add_argument("--source", choices=["adl", "epic", "both"], default="both")
    args = parser.parse_args()

    all_records: list[dict] = []

    if args.source in ("adl", "both"):
        adl_records = extract_adl_frames(dry_run=args.dry_run)
        print(f"[ADL] {len(adl_records)} frames")
        all_records.extend(adl_records)

    if args.source in ("epic", "both"):
        epic_records = extract_epic_frames(dry_run=args.dry_run)
        print(f"[EPIC] {len(epic_records)} frames")
        all_records.extend(epic_records)

    write_manifest(all_records, append=False)
    print(f"[done] total {len(all_records)} frame records")


if __name__ == "__main__":
    main()
