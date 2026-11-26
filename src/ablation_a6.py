import json
import numpy as np
from sklearn.metrics import f1_score
from baselines import run_b4
from config import BASELINES_DEV_SAMPLES, BASE_MODEL, BENCH_DIR, CHUNKING_STRATEGY, CORPUS_DIR, RES_DIR, SEED
from eval_baselines import score_preds, select_stratified
from ir_metrics import load_jsonl
from retrieval import RetrievalEngine, apply_chunking, load_corpus


def classify_query_slice(q):
    if not q["is_answerable"]:
        return "no_evidence"
    low = q["text"].lower()
    if any(w in low for w in ("tapi", "namun", "cuma", "sayang", "padahal", "walaupun", "meskipun")):
        return "conflict"
    if any(w in low for w in ("bgt", "yg", "ga", "gak", "gk", "tdk", "udh", "sdh", "blm", "bangett", "mantap", "nyesel")):
        return "ambiguous_multi_hop"
    return "easy"


def evaluate_slices(items, preds):
    slices = {"easy": [], "ambiguous_multi_hop": [], "conflict": [], "no_evidence": []}
    for q, p in zip(items, preds):
        sl = classify_query_slice(q)
        slices[sl].append((q, p))

    out = {}
    for sl_name, pairs in slices.items():
        if not pairs:
            out[sl_name] = {"samples": 0, "macro_f1_emotion": 0.0, "accuracy_emotion": 0.0}
            continue
        yt = [q["gold_emotion"] for q, _ in pairs]
        yp = [p["emotion"] for _, p in pairs]
        acc = sum(a == b for a, b in zip(yt, yp)) / len(yt)
        f1 = float(f1_score(yt, yp, average="macro", zero_division=0))
        out[sl_name] = {
            "samples": len(pairs),
            "macro_f1_emotion": round(f1, 4),
            "accuracy_emotion": round(acc, 4),
        }
    return out


def main():
    RES_DIR.mkdir(parents=True, exist_ok=True)
    docs = apply_chunking(load_corpus(CORPUS_DIR / "corpus.jsonl"), CHUNKING_STRATEGY)
    eng = RetrievalEngine(docs)
    dev_items = select_stratified(load_jsonl(BENCH_DIR / "dev_benchmark_120.jsonl"), BASELINES_DEV_SAMPLES)

    policies = {
        "fixed_retrieval_policy": False,
        "adaptive_retrieval_policy": True,
    }

    results = {
        "split": "dev",
        "variable_changed": "retrieval_policy_fixed_vs_adaptive",
        "fixed_controls": {
            "base_config": "B4",
            "model": BASE_MODEL,
            "chunking": CHUNKING_STRATEGY,
            "samples": len(dev_items),
        },
        "policies": {},
    }

    for name, is_adaptive in policies.items():
        preds = [
            run_b4(
                q["text"],
                eng,
                category=q.get("category"),
                model=BASE_MODEL,
                seed=SEED,
                use_verifier=True,
                adaptive_policy=is_adaptive,
            )
            for q in dev_items
        ]
        m = score_preds(dev_items, preds)
        avg_retries = round(float(np.mean([p.get("retry_count", 0) for p in preds])), 3)
        avg_suff = round(float(np.mean([p.get("sufficiency_score", 0.0) for p in preds])), 4)
        cost_per_f1 = round(m["avg_tokens"] / max(0.01, m["macro_f1_emotion"]), 1)

        results["policies"][name] = {
            **m,
            "avg_retries": avg_retries,
            "avg_sufficiency_score": avg_suff,
            "tokens_per_f1_unit": cost_per_f1,
            "slices": evaluate_slices(dev_items, preds),
        }

    results["selected"] = "adaptive_retrieval_policy"
    out_path = RES_DIR / "ablation_a6.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
