import json
import numpy as np
from sklearn.metrics import f1_score
from baselines import run_b4
from config import BASELINES_DEV_SAMPLES, BASE_MODEL, BENCH_DIR, CHUNKING_STRATEGY, CORPUS_DIR, RES_DIR, SEED
from eval_baselines import select_stratified
from ir_metrics import load_jsonl
from retrieval import RetrievalEngine, apply_chunking, load_corpus


def evaluate_a3_variant(items, preds):
    y_true_e, y_pred_e = [], []
    y_true_s, y_pred_s = [], []
    unsupported_cnt = 0
    false_refusal_cnt = 0
    missed_refusal_cnt = 0
    ans_cnt = 0
    unans_cnt = 0
    span_ok = 0
    lats, toks = [], []

    for q, p in zip(items, preds):
        y_true_e.append(q["gold_emotion"])
        y_pred_e.append(p["emotion"])
        y_true_s.append(q["gold_sentiment"])
        y_pred_s.append(p["sentiment"])
        lats.append(p["latency_ms"])
        toks.append(p["tokens"])

        spans = p.get("evidence_spans", [])
        if isinstance(spans, str):
            spans = [spans] if spans.strip() else []
        text_low = q["text"].lower()
        all_verbatim = bool(spans) and all(s.lower() in text_low for s in spans)

        if q["is_answerable"]:
            ans_cnt += 1
            if p.get("abstain") or p.get("emotion") == "ABSTAIN":
                false_refusal_cnt += 1
            if not all_verbatim:
                unsupported_cnt += 1
            else:
                span_ok += 1
        else:
            unans_cnt += 1
            if not p.get("abstain") and p.get("emotion") != "ABSTAIN":
                missed_refusal_cnt += 1
                unsupported_cnt += 1
            else:
                span_ok += 1

    return {
        "macro_f1_emotion": round(float(f1_score(y_true_e, y_pred_e, average="macro", zero_division=0)), 4),
        "macro_f1_sentiment": round(float(f1_score(y_true_s, y_pred_s, average="macro", zero_division=0)), 4),
        "evidence_support": round(span_ok / max(1, len(items)), 4),
        "unsupported_claim_rate": round(unsupported_cnt / max(1, len(items)), 4),
        "false_refusal_rate": round(false_refusal_cnt / max(1, ans_cnt), 4),
        "missed_refusal_rate": round(missed_refusal_cnt / max(1, unans_cnt), 4),
        "p95_latency_ms": round(float(np.percentile(lats, 95)), 1),
        "avg_tokens": round(float(np.mean(toks)), 1),
        "samples": len(items),
    }


def main():
    RES_DIR.mkdir(parents=True, exist_ok=True)
    docs = apply_chunking(load_corpus(CORPUS_DIR / "corpus.jsonl"), CHUNKING_STRATEGY)
    eng = RetrievalEngine(docs)
    dev_items = select_stratified(load_jsonl(BENCH_DIR / "dev_benchmark_120.jsonl"), BASELINES_DEV_SAMPLES)

    variants = {
        "no_verifier": False,
        "independent_verifier_bounded_repair": True,
    }

    results = {
        "split": "dev",
        "variable_changed": "claim_and_span_verifier",
        "fixed_controls": {
            "base_config": "B4",
            "model": BASE_MODEL,
            "chunking": CHUNKING_STRATEGY,
            "retriever": "hybrid_rerank",
            "samples": len(dev_items),
        },
        "definitions": {
            "unsupported_claim_rate": "Proporsi output yang mengutip evidence_spans dari luar teks ulasan (halusinasi konteks) atau gagal menyertakan bukti valid.",
            "false_refusal_rate": "Proporsi kueri answerable yang salah ditolak (abstain=true) oleh model.",
        },
        "variants": {},
    }

    for name, use_ver in variants.items():
        preds = [
            run_b4(
                q["text"],
                eng,
                category=q.get("category"),
                model=BASE_MODEL,
                seed=SEED,
                use_verifier=use_ver,
                adaptive_policy=True,
            )
            for q in dev_items
        ]
        results["variants"][name] = evaluate_a3_variant(dev_items, preds)

    results["selected"] = "independent_verifier_bounded_repair"
    out_path = RES_DIR / "ablation_a3.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
