import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from baselines import run_b3
from config import (
    A7_HIGH_CHARS,
    A7_HIGH_TOPK,
    A7_LOW_CHARS,
    A7_LOW_TOPK,
    A7_SAMPLES,
    BASE_MODEL,
    CHUNKING_STRATEGY,
    CORPUS_DIR,
    PROC_DIR,
    RES_DIR,
    SEED,
)
from retrieval import RetrievalEngine, apply_chunking, load_corpus

def main():
    RES_DIR.mkdir(parents=True, exist_ok=True)
    docs = apply_chunking(load_corpus(CORPUS_DIR / "corpus.jsonl"), CHUNKING_STRATEGY)
    raw_dev = pd.read_csv(PROC_DIR / "dev.csv")
    per_cls = max(1, A7_SAMPLES // 5)
    dev_df = pd.concat([g.sample(n=min(per_cls, len(g)), random_state=SEED) for _, g in raw_dev.groupby("Emotion")]).reset_index(drop=True)
    eng = RetrievalEngine(docs)

    budgets = {
        "low_budget": {"top_k": A7_LOW_TOPK, "max_ctx_chars": A7_LOW_CHARS},
        "high_budget": {"top_k": A7_HIGH_TOPK, "max_ctx_chars": A7_HIGH_CHARS},
    }

    results = {"split": "dev", "model": BASE_MODEL, "chunking": CHUNKING_STRATEGY, "samples": len(dev_df), "budgets": {}}
    for name, cfg in budgets.items():
        y_true, y_pred, lats, toks = [], [], [], []
        for _, row in dev_df.iterrows():
            res = run_b3(
                str(row["Customer Review"]),
                eng,
                category=str(row["Category"]),
                top_k=cfg["top_k"],
                max_ctx_chars=cfg["max_ctx_chars"],
            )
            y_true.append(str(row["Emotion"]))
            y_pred.append(res["emotion"])
            lats.append(res["latency_ms"])
            toks.append(res["tokens"])

        results["budgets"][name] = {
            "top_k": cfg["top_k"],
            "max_ctx_chars": cfg["max_ctx_chars"],
            "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
            "p95_latency_ms": round(float(np.percentile(lats, 95)), 1),
            "avg_tokens": round(float(np.mean(toks)), 1),
        }

    best = max(results["budgets"], key=lambda k: results["budgets"][k]["macro_f1"])
    results["selected"] = best

    out_path = RES_DIR / "ablation_a7.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
