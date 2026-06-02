#!/usr/bin/env python3
"""Download VisDrone MOT datasets (assignment val + train for fine-tuning)."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

# Google Drive file IDs (VisDrone community mirrors)
GDRIVE = {
    "mot_val_assignment": "1rqnKe9IgU_crMaxRoel9_nuUsMEBBVQu",  # Task 4 MOT val (assignment link)
    "mot_train": "1Cc1vVmqExJKyFO1VNSGgPJuNZ3PKlz9e",  # VisDrone2019-MOT-train
}


def download_gdrive(file_id: str, dest: Path) -> Path:
    import gdown

    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    out = dest if dest.suffix else dest / f"{file_id}.zip"
    gdown.download(url, str(out), quiet=False)
    return out


def extract_zip(zip_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    return out_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    root = args.data_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)

    print("Downloading MOT validation set (assignment)...")
    val_zip = download_gdrive(GDRIVE["mot_val_assignment"], root / "visdrone_mot_val.zip")
    val_dir = extract_zip(val_zip, root / "mot_val")
    print(f"Val extracted to {val_dir}")

    if not args.skip_train:
        print("Downloading MOT train set (for fine-tuning)...")
        train_zip = download_gdrive(GDRIVE["mot_train"], root / "visdrone_mot_train.zip")
        train_dir = extract_zip(train_zip, root / "mot_train")
        print(f"Train extracted to {train_dir}")

    print("Done. Run: python scripts/prepare_dataset.py")


if __name__ == "__main__":
    main()
