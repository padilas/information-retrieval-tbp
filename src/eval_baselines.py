import argparse
import json
import numpy as np
from sklearn.metrics import f1_score
from baselines import run_b0, run_b1, run_b2, run_b3, run_b4, run_b5, run_b6
from config import BASELINES_DEV_SAMPLES, BASE_MODEL, BENCH_DIR, CHUNKING_STRATEGY, CORPUS_DIR, RES_DIR, SEED
from ir_metrics import load_jsonl
from retrieval import RetrievalEngine, apply_chunking, chunk_fixed, load_corpus

def select_stratified(items, limit):
    if limit >= len(items):
        return items
    ans = [x for x in items if x["is_answerable"]]
    unans = [x for x in items if not x["is_answerable"]]
    n_unans = max(2, round(limit * (len(unans) / len(items))))
    n_ans = max(5, limit - n_unans)

    by_emo = {}
    for x in ans:
        by_emo.setdefault(x["gold_emotion"], []).append(x)
    per_emo = max(1, n_ans // max(1, len(by_emo)))
    picked = []
    for grp in by_emo.values():
        picked.extend(grp[:per_emo])
    picked.extend(unans[:n_unans])
    return picked[:limit]

def score_preds(items, preds):
    y_true_s, y_pred_s = [], []
    y_true_e, y_pred_e = [], []
    span_ok, lats, toks = 0, [], []

    for q, p in zip(items, preds):
        y_true_s.append(q["gold_sentiment"])
        y_pred_s.append(p["sentiment"])
        y_true_e.append(q["gold_emotion"])
        y_pred_e.append(p["emotion"])
        lats.append(p["latency_ms"])
        toks.append(p["tokens"])

        spans = p.get("evidence_spans", [])
        if isinstance(spans, str):
            spans = [spans] if spans.strip() else []
        text_low = q["text"].lower()
        if not q["is_answerable"] and p.get("abstain"):
            span_ok += 1
        elif q["is_answerable"] and spans and all(s.lower() in text_low for s in spans):
            span_ok += 1

    return {
        "macro_f1_emotion": round(float(f1_score(y_true_e, y_pred_e, average="macro", zero_division=0)), 4),
        "macro_f1_sentiment": round(float(f1_score(y_true_s, y_pred_s, average="macro", zero_division=0)), 4),
        "evidence_support": round(span_ok / max(1, len(items)), 4),
        "p95_latency_ms": round(float(np.percentile(lats, 95)), 1),
        "avg_tokens": round(float(np.mean(toks)), 1),
        "samples": len(items),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--limit", type=int, default=BASELINES_DEV_SAMPLES)
    parser.add_argument("--recompute-all", action="store_true", help="Hitung ulang B0-B3 dari awal")
    args = parser.parse_args()

    RES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RES_DIR / f"baselines_{args.split}.json"
    existing = {}
    if out_path.exists() and not args.recompute_all:
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = {}

    raw_docs = load_corpus(CORPUS_DIR / "corpus.jsonl")
    fixed_eng = RetrievalEngine(chunk_fixed(raw_docs))
    best_eng = RetrievalEngine(apply_chunking(raw_docs, CHUNKING_STRATEGY))

    bench_file = "dev_benchmark_120.jsonl" if args.split == "dev" else "gold_test.jsonl"
    items = select_stratified(load_jsonl(BENCH_DIR / bench_file), args.limit)

    runners = {
        "B0": lambda q: run_b0(q["text"], model=BASE_MODEL, seed=SEED),
        "B1": lambda q: run_b1(q["text"], model=BASE_MODEL, seed=SEED),
        "B2": lambda q: run_b2(q["text"], fixed_eng, model=BASE_MODEL, seed=SEED),
        "B3": lambda q: run_b3(q["text"], best_eng, category=q.get("category"), model=BASE_MODEL, seed=SEED),
        "B4": lambda q: run_b4(q["text"], best_eng, category=q.get("category"), model=BASE_MODEL, seed=SEED),
        "B5": lambda q: run_b5(q["text"], best_eng, category=q.get("category"), model=BASE_MODEL, seed=SEED),
        "B6": lambda q: run_b6(q["text"], best_eng, category=q.get("category"), model=BASE_MODEL, seed=SEED),
    }

    summary = {
        "split": args.split,
        "model": BASE_MODEL,
        "chunking": CHUNKING_STRATEGY,
        "metrics": dict(existing.get("metrics", {})),
        "outputs": dict(existing.get("outputs", {})),
    }
    for name, fn in runners.items():
        if name in ("B0", "B1", "B2", "B3") and name in summary["metrics"] and name in summary["outputs"]:
            print(name, "(preserved)", summary["metrics"][name])
            continue
        preds = [fn(q) for q in items]
        summary["metrics"][name] = score_preds(items, preds)
        summary["outputs"][name] = [{"query_id": q["query_id"], **p} for q, p in zip(items, preds)]
        print(name, summary["metrics"][name])
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
