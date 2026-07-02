"""
Downloads snake images from iNaturalist (free, no auth needed) and optionally
from Kaggle. Organises them into data/raw/venomous/ and data/raw/non_venomous/.

Usage:
    python download_data.py                    # iNaturalist only (default)
    python download_data.py --kaggle           # also download Kaggle datasets
    python download_data.py --max 150          # 150 images per taxon
"""
import argparse
import io
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

DATA_DIR = Path("data/raw")
VENOMOUS_DIR = DATA_DIR / "venomous"
NON_VENOMOUS_DIR = DATA_DIR / "non_venomous"

INATURALIST_API = "https://api.inaturalist.org/v1"
DOWNLOAD_WORKERS = 6
API_DELAY = 0.4          # seconds between API calls to respect rate limits

# ── Taxonomy ───────────────────────────────────────────────────────────────────
VENOMOUS_TAXA = [
    "Crotalus",       # Rattlesnakes (Americas)
    "Sistrurus",      # Pygmy rattlesnakes / massasaugas
    "Agkistrodon",    # Copperheads & Cottonmouths
    "Naja",           # Cobras (Asia/Africa)
    "Ophiophagus",    # King Cobra
    "Bungarus",       # Kraits (Asia)
    "Dendroaspis",    # Mambas (Africa)
    "Bitis",          # Puff adders & Gaboon vipers
    "Cerastes",       # Horned vipers
    "Vipera",         # European vipers
    "Bothrops",       # Lancehead vipers (South America)
    "Lachesis",       # Bushmasters
    "Micrurus",       # Coral snakes (Americas)
    "Oxyuranus",      # Taipans (Australia)
    "Pseudonaja",     # Brown snakes (Australia)
    "Notechis",       # Tiger snakes (Australia)
    "Acanthophis",    # Death adders (Australia/NG)
    "Echis",          # Saw-scaled vipers (Africa/Asia)
]

NON_VENOMOUS_TAXA = [
    "Python",         # Pythons (Asia/Africa/Australia)
    "Boa",            # Boa constrictors (Americas)
    "Lampropeltis",   # King snakes & milk snakes
    "Pantherophis",   # Corn snakes & rat snakes
    "Thamnophis",     # Garter snakes
    "Pituophis",      # Bull snakes & pine snakes
    "Coluber",        # Racers
    "Morelia",        # Carpet pythons (Australia)
    "Elaphe",         # Eurasian rat snakes
    "Diadophis",      # Ring-neck snakes
    "Heterodon",      # Hognose snakes
    "Nerodia",        # Water snakes (non-venomous, N. America)
    "Liasis",         # Water pythons (Australia)
    "Eryx",           # Sand boas (Old World)
    "Storeria",       # Brown snakes (harmless, N. America)
    "Virginia",       # Earth snakes
    "Charina",        # Rubber boas
]

# Kaggle datasets to optionally download (requires `kaggle` CLI configured)
KAGGLE_DATASETS = [
    "adityasharma01/snake-dataset-india",
    "shekhar234/snake-dataset",
]

# ── iNaturalist downloader ─────────────────────────────────────────────────────

def fetch_photo_urls(taxon_name: str, max_count: int) -> list[str]:
    """Return up to max_count medium-resolution photo URLs for the taxon."""
    urls = []
    page = 1

    while len(urls) < max_count:
        try:
            resp = requests.get(
                f"{INATURALIST_API}/observations",
                params={
                    "taxon_name": taxon_name,
                    "quality_grade": "research",
                    "photos": True,
                    "per_page": 200,
                    "page": page,
                    "order": "desc",
                    "order_by": "votes",
                    "photo_license": "cc-by,cc-by-nc,cc-by-sa,cc0",
                },
                timeout=20,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            if not results:
                break

            for obs in results:
                for photo in obs.get("photos", []):
                    raw_url = photo.get("url", "")
                    if raw_url:
                        urls.append(raw_url.replace("square", "medium"))
                        if len(urls) >= max_count:
                            break
                if len(urls) >= max_count:
                    break

            if len(results) < 200:
                break
            page += 1
            time.sleep(API_DELAY)

        except Exception as exc:
            print(f"  API error for {taxon_name} (page {page}): {exc}")
            break

    return urls[:max_count]


def _download_one(args: tuple) -> bool:
    url, path = args
    if path.exists():
        return True
    try:
        resp = requests.get(url, timeout=20, stream=True)
        resp.raise_for_status()
        content = resp.content
        img = Image.open(io.BytesIO(content))
        img.verify()
        img = Image.open(io.BytesIO(content)).convert("RGB")
        if min(img.size) < 80:
            return False
        img.save(path, "JPEG", quality=88)
        return True
    except Exception:
        return False


def download_taxon(taxon_name: str, save_dir: Path, max_images: int) -> int:
    save_dir.mkdir(parents=True, exist_ok=True)
    existing = sum(1 for _ in save_dir.glob("*.jpg"))

    if existing >= max_images:
        print(f"  {taxon_name}: already have {existing} images — skipping")
        return existing

    needed = max_images - existing
    print(f"  {taxon_name}: fetching up to {needed} more images…")
    urls = fetch_photo_urls(taxon_name, needed)

    if not urls:
        print(f"  {taxon_name}: no images found on iNaturalist")
        return existing

    tasks = [(url, save_dir / f"{taxon_name.lower()}_{i:05d}.jpg")
             for i, url in enumerate(urls, start=existing)]

    ok = 0
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        futs = {pool.submit(_download_one, t): t for t in tasks}
        for fut in tqdm(as_completed(futs), total=len(tasks), desc=taxon_name, leave=False, ncols=80):
            if fut.result():
                ok += 1

    total = existing + ok
    print(f"  {taxon_name}: {ok} new images  (total {total})")
    return total


# ── Kaggle importer ────────────────────────────────────────────────────────────

def _copy_tree(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        for img in src.rglob(ext):
            shutil.copy2(img, dst / img.name)
            copied += 1
    return copied


def import_kaggle_opendatasets():
    """Download Kaggle datasets via opendatasets (prompts for credentials once)."""
    try:
        import opendatasets as od
    except ImportError:
        print("  opendatasets not installed — skipping Kaggle download")
        return

    for dataset_url in [
        "https://www.kaggle.com/datasets/adityasharma01/snake-dataset-india",
        "https://www.kaggle.com/datasets/shekhar234/snake-dataset",
    ]:
        name = dataset_url.rstrip("/").split("/")[-1]
        print(f"\n  Downloading from Kaggle: {name}")
        try:
            od.download(dataset_url, data_dir="data/kaggle", force=False)
        except Exception as exc:
            print(f"  Failed: {exc}")
            continue

        # Try to find and copy organised images
        for kaggle_dir in Path("data/kaggle").rglob("*"):
            if not kaggle_dir.is_dir():
                continue
            dname = kaggle_dir.name.lower().replace(" ", "_")
            if "venomous" in dname and "non" not in dname:
                n = _copy_tree(kaggle_dir, VENOMOUS_DIR / f"kaggle_{name}")
                print(f"  Copied {n} venomous images from {kaggle_dir}")
            elif "non" in dname and "venomous" in dname:
                n = _copy_tree(kaggle_dir, NON_VENOMOUS_DIR / f"kaggle_{name}")
                print(f"  Copied {n} non-venomous images from {kaggle_dir}")


def auto_import_existing_kaggle():
    """Import the Kaggle dataset already downloaded by the notebook."""
    candidates = [
        Path("snake-dataset-india") / "Snake Images",
        Path("../snake-dataset-india") / "Snake Images",
        Path("data/kaggle/snake-dataset-india") / "Snake Images",
    ]
    for base in candidates:
        if not base.exists():
            continue
        print(f"\n  Found existing Kaggle dataset at {base}")
        for split in ("train", "test"):
            v_src = base / split / "Venomous"
            nv_src = base / split / "Non Venomous"
            if v_src.exists():
                n = _copy_tree(v_src, VENOMOUS_DIR / "kaggle_india")
                print(f"  Copied {n} venomous images from {split}")
            if nv_src.exists():
                n = _copy_tree(nv_src, NON_VENOMOUS_DIR / "kaggle_india")
                print(f"  Copied {n} non-venomous images from {split}")
        return True
    return False


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download snake image dataset")
    parser.add_argument("--max", type=int, default=300,
                        help="Max images per taxon from iNaturalist (default 300)")
    parser.add_argument("--kaggle", action="store_true",
                        help="Also download from Kaggle (requires credentials)")
    args = parser.parse_args()

    print("=" * 60)
    print("   Snake Dataset Downloader")
    print(f"   Target: {args.max} images per taxon from iNaturalist")
    print("=" * 60)

    # Import existing Kaggle data if present
    auto_import_existing_kaggle()

    if args.kaggle:
        print("\n[Kaggle] Downloading additional datasets…")
        import_kaggle_opendatasets()

    print("\n[1/2] VENOMOUS snake images")
    venomous_total = 0
    for taxon in VENOMOUS_TAXA:
        venomous_total += download_taxon(taxon, VENOMOUS_DIR / taxon.lower(), args.max)
        time.sleep(0.5)

    print("\n[2/2] NON-VENOMOUS snake images")
    non_venomous_total = 0
    for taxon in NON_VENOMOUS_TAXA:
        non_venomous_total += download_taxon(taxon, NON_VENOMOUS_DIR / taxon.lower(), args.max)
        time.sleep(0.5)

    print("\n" + "=" * 60)
    print(f"  Venomous images    : {venomous_total:,}")
    print(f"  Non-venomous images: {non_venomous_total:,}")
    print(f"  Total              : {venomous_total + non_venomous_total:,}")
    print("=" * 60)
    print("\nDone!  Next step:  python train.py")


if __name__ == "__main__":
    main()
