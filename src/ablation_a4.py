import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from advanced_rag import get_or_train_lora_adapter
from baselines import run_b4, run_b5
from config import B5_LORA_ADAPTER_DIR, BASELINES_DEV_SAMPLES, BASE_MODEL, BENCH_DIR, CHUNKING_STRATEGY, CORPUS_DIR, PROC_DIR, RES_DIR, SEED
from eval_baselines import score_preds, select_stratified
from ir_metrics import load_jsonl
from retrieval import RetrievalEngine, apply_chunking, load_corpus


def load_ood_slice(limit=24):
    df = pd.read_csv(PROC_DIR / "ood_test.csv")
    per_emo = max(1, limit // max(1, df["Emotion"].nunique()))
    sampled = pd.concat(
        [g.sample(n=min(per_emo, len(g)), random_state=SEED) for _, g in df.groupby("Emotion")]
    ).reset_index(drop=True)
    if len(sampled) < limit:
        rem = df.drop(sampled.index, errors="ignore").head(limit - len(sampled))
        sampled = pd.concat([sampled, rem]).reset_index(drop=True)
    items = []
    for idx, row in sampled.head(limit).iterrows():
        items.append({
            "query_id": f"OOD_{idx+1:03d}",
            "text": str(row["Customer Review"]),
            "category": str(row["Category"]),
            "gold_sentiment": str(row["Sentiment"]),
            "gold_emotion": str(row["Emotion"]),
            "is_answerable": True,
            "slice": "ood_category_drift",
        })
    return items


def main():
    RES_DIR.mkdir(parents=True, exist_ok=True)
    docs = apply_chunking(load_corpus(CORPUS_DIR / "corpus.jsonl"), CHUNKING_STRATEGY)
    eng = RetrievalEngine(docs)

    lora_bundle = get_or_train_lora_adapter()
    train_log = lora_bundle["train_log"]

    in_domain_items = select_stratified(load_jsonl(BENCH_DIR / "dev_benchmark_120.jsonl"), BASELINES_DEV_SAMPLES)
    ood_items = load_ood_slice(limit=BASELINES_DEV_SAMPLES)

    b4_in_preds = [run_b4(q["text"], eng, category=q.get("category"), model=BASE_MODEL, seed=SEED) for q in in_domain_items]
    b5_in_preds = [run_b5(q["text"], eng, category=q.get("category"), model=BASE_MODEL, seed=SEED) for q in in_domain_items]

    b4_ood_preds = [run_b4(q["text"], eng, category=q.get("category"), model=BASE_MODEL, seed=SEED) for q in ood_items]
    b5_ood_preds = [run_b5(q["text"], eng, category=q.get("category"), model=BASE_MODEL, seed=SEED) for q in ood_items]

    m_b4_in = score_preds(in_domain_items, b4_in_preds)
    m_b5_in = score_preds(in_domain_items, b5_in_preds)
    m_b4_ood = score_preds(ood_items, b4_ood_preds)
    m_b5_ood = score_preds(ood_items, b5_ood_preds)

    head_file = B5_LORA_ADAPTER_DIR / "lora_domain_head.pkl"
    adapter_kb = round(head_file.stat().st_size / 1024.0, 1) if head_file.exists() else 0.0

    results = {
        "split": "dev_and_ood",
        "variable_changed": "peft_lora_adapter",
        "fixed_controls": {
            "parent_config": "B4",
            "base_model": BASE_MODEL,
            "chunking": CHUNKING_STRATEGY,
            "retriever": "hybrid_rerank",
            "in_domain_samples": len(in_domain_items),
            "ood_samples": len(ood_items),
            "ood_categories": sorted(list({q["category"] for q in ood_items})),
        },
        "variants": {
            "no_adapter_b4": {
                "in_domain": m_b4_in,
                "ood_category_drift": m_b4_ood,
            },
            "peft_lora_b5": {
                "in_domain": m_b5_in,
                "ood_category_drift": m_b5_ood,
            },
        },
        "analysis": {
            "in_domain_f1_emotion_gain": round(m_b5_in["macro_f1_emotion"] - m_b4_in["macro_f1_emotion"], 4),
            "ood_f1_emotion_gain": round(m_b5_ood["macro_f1_emotion"] - m_b4_ood["macro_f1_emotion"], 4),
            "train_emotion_accuracy": train_log["train_emotion_accuracy"],
            "overfit_gap_train_vs_dev": round(train_log["train_emotion_accuracy"] - m_b5_in["macro_f1_emotion"], 4),
            "resource_overhead": {
                "adapter_size_kb": adapter_kb,
                "trainable_params_pct": train_log["trainable_params_pct"],
                "training_duration_sec": train_log["training_duration_sec"],
                "peak_vram_mb": train_log["peak_memory_mb"],
            },
        },
        "selected": "peft_lora_b5",
    }

    out_path = RES_DIR / "ablation_a4.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
