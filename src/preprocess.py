import hashlib
import json
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from config import BENCH_DIR, PROC_DIR, PROJECT_ROOT, RAW_CSV, SEED

OOD_CATS = ["Precious Metal", "Tour and Travel", "Property"]

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def clean_surface(text):
    s = str(text).replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\?{2,}", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def alnum_key(text):
    return re.sub(r"[^a-z0-9]", "", str(text).lower())

def load_locked_test_keys():
    ann_path = BENCH_DIR / "annotator_1.csv"
    if not ann_path.exists():
        return set()
    df = pd.read_csv(ann_path, encoding="utf-8-sig").fillna("")
    if "emotion" not in df.columns or (df["emotion"].astype(str).str.strip() == "").all():
        return set()
    return {alnum_key(t) for t in df["text"] if alnum_key(t)}

def main():
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    raw_df = pd.read_csv(RAW_CSV)
    raw_hash = sha256_file(RAW_CSV)

    df = raw_df.copy()
    df["Customer Review"] = df["Customer Review"].map(clean_surface)
    df["_key"] = df["Customer Review"].map(alnum_key)
    df = df[df["_key"].str.len() >= 2].copy()

    dups = df[df.duplicated("_key", keep=False)]
    conflicts = []
    resolved = {}
    for key, grp in dups.groupby("_key"):
        emos = grp["Emotion"].unique().tolist()
        if len(emos) > 1:
            chosen = grp["Emotion"].mode().iloc[0]
            resolved[key] = chosen
            conflicts.append({
                "text": grp["Customer Review"].iloc[0],
                "emotions": emos,
                "resolved": chosen,
                "count": int(len(grp)),
            })

    with open(PROC_DIR / "conflict_log.json", "w", encoding="utf-8") as f:
        json.dump(conflicts, f, indent=2, ensure_ascii=False)

    clean = df.drop_duplicates(subset=["_key"], keep="first").copy()
    if resolved:
        mask = clean["_key"].isin(resolved)
        clean.loc[mask, "Emotion"] = clean.loc[mask, "_key"].map(resolved)

    locked_keys = load_locked_test_keys()
    locked_df = clean[clean["_key"].isin(locked_keys)].drop(columns=["_key"]).copy()
    rem_clean = clean[~clean["_key"].isin(locked_keys)].drop(columns=["_key"]).copy()

    ood = rem_clean[rem_clean["Category"].isin(OOD_CATS)].copy()
    indomain = rem_clean[~rem_clean["Category"].isin(OOD_CATS)].copy()

    train_df, hold_df = train_test_split(indomain, test_size=0.22, random_state=SEED, stratify=indomain["Emotion"])
    dev_df, extra_pool = train_test_split(hold_df, test_size=0.30, random_state=SEED, stratify=hold_df["Emotion"])
    test_pool = pd.concat([locked_df, extra_pool], ignore_index=True)

    train_path = PROC_DIR / "train.csv"
    dev_path = PROC_DIR / "dev.csv"
    pool_path = PROC_DIR / "test_pool.csv"
    ood_path = PROC_DIR / "ood_test.csv"

    train_df.to_csv(train_path, index=False)
    dev_df.to_csv(dev_path, index=False)
    test_pool.to_csv(pool_path, index=False)
    ood.to_csv(ood_path, index=False)

    manifest = {
        "raw": {"path": RAW_CSV.relative_to(PROJECT_ROOT).as_posix(), "rows": len(raw_df), "sha256": raw_hash},
        "dedup_rows": len(clean),
        "conflicts_resolved": len(conflicts),
        "locked_test_excluded_from_train_dev": len(locked_df),
        "splits": {
            "train": {"path": train_path.relative_to(PROJECT_ROOT).as_posix(), "rows": len(train_df), "sha256": sha256_file(train_path)},
            "dev": {"path": dev_path.relative_to(PROJECT_ROOT).as_posix(), "rows": len(dev_df), "sha256": sha256_file(dev_path)},
            "test_pool": {"path": pool_path.relative_to(PROJECT_ROOT).as_posix(), "rows": len(test_pool), "sha256": sha256_file(pool_path)},
            "ood_test": {"path": ood_path.relative_to(PROJECT_ROOT).as_posix(), "rows": len(ood), "sha256": sha256_file(ood_path)},
        },
    }
    with open(PROC_DIR / "manifest_splits.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    source_reg = {
        "dataset": "PRDECT-ID",
        "doi": "10.17632/574v66hf2v.1",
        "url": "https://data.mendeley.com/datasets/574v66hf2v/1",
        "repo": "https://github.com/rhiosutoyo/PRDECT-ID-Indonesian-Product-Reviews-Dataset",
        "license": "CC BY 4.0",
        "citation": "Sutoyo et al. (2022). PRDECT-ID: Indonesian product reviews dataset for emotions classification tasks. Data in Brief, 44, 108554.",
    }
    with open(PROC_DIR / "source_register.json", "w", encoding="utf-8") as f:
        json.dump(source_reg, f, indent=2)

    print(f"dedup={len(clean)} locked_test={len(locked_df)} train={len(train_df)} dev={len(dev_df)} test_pool={len(test_pool)} ood={len(ood)}")

if __name__ == "__main__":
    main()
