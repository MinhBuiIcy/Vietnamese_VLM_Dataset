"""Filter COCO images by indoor categories, augment, and append to manifest.

Indoor categories: chair, couch, laptop, cup, bottle, dining table, bed,
book, clock, vase, bowl, keyboard, mouse, remote, cell phone, microwave,
oven, sink, refrigerator

Outputs:
    OUTPUT_DIR/coco_augmented/  — augmented JPEG images
    data/frames_manifest.jsonl  — appended with COCO entries
"""

import argparse
import json
import os
import random
from pathlib import Path

import albumentations as A
import cv2
from dotenv import load_dotenv
from pycocotools.coco import COCO
from tqdm import tqdm

load_dotenv()

COCO_DIR = Path(os.getenv("COCO_DIR", ""))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))
MANIFEST_PATH = Path("data/frames_manifest.jsonl")

INDOOR_CATEGORY_NAMES = {
    "chair", "couch", "laptop", "cup", "bottle", "dining table",
    "bed", "book", "clock", "vase", "bowl", "keyboard", "mouse",
    "remote", "cell phone", "microwave", "oven", "sink", "refrigerator",
}

AUGMENT_PIPELINE = A.Compose([
    A.RandomCrop(height=400, width=400, pad_if_needed=True),
    A.MotionBlur(blur_limit=(3, 7), p=0.3),
    A.RandomBrightnessContrast(p=0.4),
    A.GaussNoise(p=0.3),
    A.Rotate(limit=15, p=0.4),
])


def load_coco_indoor_images(ann_file: Path, images_dir: Path) -> list[dict]:
    """Return list of {image_id, file_name, has_person} for indoor images."""
    coco = COCO(str(ann_file))

    # Map category name → id
    cats = coco.loadCats(coco.getCatIds())
    name_to_id = {c["name"]: c["id"] for c in cats}

    indoor_ids = {name_to_id[n] for n in INDOOR_CATEGORY_NAMES if n in name_to_id}
    person_id = name_to_id.get("person")

    # Images that have ≥1 indoor category annotation
    indoor_img_ids: set[int] = set()
    for cat_id in indoor_ids:
        indoor_img_ids.update(coco.getImgIds(catIds=[cat_id]))

    # Build per-image person presence
    person_img_ids: set[int] = set()
    if person_id is not None:
        person_img_ids = set(coco.getImgIds(catIds=[person_id]))

    images = coco.loadImgs(list(indoor_img_ids))
    result = []
    for img in images:
        fp = images_dir / img["file_name"]
        if fp.exists():
            result.append({
                "image_id": img["id"],
                "file_name": img["file_name"],
                "full_path": fp,
                "has_person": img["id"] in person_img_ids,
            })
    return result


def augment_and_save(image_entries: list[dict], out_dir: Path, limit: int | None = None) -> list[dict]:
    """Augment images and save to out_dir. Returns manifest records."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if limit is not None:
        image_entries = random.sample(image_entries, min(limit, len(image_entries)))

    records: list[dict] = []
    for entry in tqdm(image_entries, desc="COCO augment"):
        img = cv2.imread(str(entry["full_path"]))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented = AUGMENT_PIPELINE(image=img_rgb)["image"]
        img_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)

        stem = Path(entry["file_name"]).stem
        out_name = f"coco_{stem}_aug.jpg"
        out_path = out_dir / out_name
        cv2.imwrite(str(out_path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])

        records.append({
            "frame_path": f"coco_augmented/{out_name}",
            "source": "coco",
            "has_person": entry["has_person"],
        })
    return records


def append_manifest(records: list[dict]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "a") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[manifest] appended {len(records)} COCO records → {MANIFEST_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter COCO and augment indoor images")
    parser.add_argument("--limit", type=int, default=None, help="Max images to process")
    parser.add_argument("--split", default="train2017", help="COCO split (default: train2017)")
    args = parser.parse_args()

    ann_file = COCO_DIR / "annotations" / f"instances_{args.split}.json"
    images_dir = COCO_DIR / args.split

    if not ann_file.exists():
        print(f"[COCO] Annotation file not found: {ann_file}")
        return
    if not images_dir.exists():
        print(f"[COCO] Images dir not found: {images_dir}")
        return

    print(f"[COCO] Loading annotations from {ann_file}")
    entries = load_coco_indoor_images(ann_file, images_dir)
    print(f"[COCO] {len(entries)} indoor images found")

    out_dir = OUTPUT_DIR / "coco_augmented"
    records = augment_and_save(entries, out_dir, limit=args.limit)
    print(f"[COCO] {len(records)} images augmented → {out_dir}")

    append_manifest(records)


if __name__ == "__main__":
    main()
