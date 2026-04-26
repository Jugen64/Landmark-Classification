import subprocess
from pathlib import Path
import zipfile
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import csv
import json




PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data/"

IMAGE_DIR = DATA_DIR / "processed/images"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MAPPING_DIR = DATA_DIR / "mapping.json"

import json

def save_images(dataset):
    with open(MAPPING_DIR, "r") as f:
        id_to_country = json.load(f)

    records = []

    for i, item in enumerate(dataset):
        if i % 200 == 0:
            print(f"Processed {i}")

        img = item["image"]
        img_id = item["id"]
        landmark_id = item["landmark_id"]

        country = id_to_country.get(str(landmark_id), "unknown")

        filename = f"{img_id}.jpg"
        path = IMAGE_DIR / filename

        img = img.convert("RGB")
        img.save(path, format="JPEG")

        records.append({
            "image_id": img_id,
            "landmark_id": landmark_id,
            "country": country
        })

    return records


def sans():
    print(f"Hello")
    ds = load_dataset("pemujo/GLDv2_Top_51_Categories")
    print(ds)
    print(ds["train"][0])
    records = save_images(ds["train"])
    df = pd.DataFrame(records, columns=["image_id", "landmark_id", "country"])
    df[["image_id", "landmark_id", "country"]].to_csv(DATA_DIR / "hf_output.csv", index=False)
    



if __name__ == "__main__":
    sans()