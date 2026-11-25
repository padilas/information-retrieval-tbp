import hashlib
import json
import numpy as np
from baselines import run_b3
from config import BASE_MODEL, BENCH_DIR, CHUNK_OVERLAP, CHUNK_SIZE, CORPUS_DIR, RES_DIR, SEED, TOP_K
from ir_metrics import eval_ranking, load_jsonl, load_qrels
from retrieval import RetrievalEngine, chunk_fixed, chunk_heading, chunk_hierarchical, load_corpus

def select_dev_slice(items, per_emo=3):
    by_emo = {}
    for x in items:
        if x["is_answerable"]:
            by_emo.setdefault(x["gold_emotion"], []).append(x)
    out = []
    for grp in by_emo.values():
        out.extend(grp[:per_emo])
    return out

def measure_llm_groundedness(eng, sample_queries):
    grounded_cnt = 0
    for q in sample_queries:
        res = run_b3(q["text"], eng, category=q.get("category"), top_k=TOP_K, model=BASE_MODEL, seed=SEED)
        spans = res.get("evidence_spans", [])
        cites = set(res.get("policy_citations", []))
        ret_ids = set(res.get("retrieved_ids", []))
        text_low = q["text"].lower()
        span_verbatim = bool(spans) and all(s.lower() in text_low for s in spans)
        cite_valid = bool(cites) and cites.issubset(ret_ids)
        if span_verbatim and cite_valid and not res.get("abstain"):
            grounded_cnt += 1
    return round(grounded_cnt / max(1, len(sample_queries)), 4)

def main():
    RES_DIR.mkdir(parents=True, exist_ok=True)
    docs = load_corpus(CORPUS_DIR / "corpus.jsonl")
    dev_bench = load_jsonl(BENCH_DIR / "dev_benchmark_120.jsonl")
    qrels_path = BENCH_DIR / "qrels_dev.tsv"
    qrels = load_qrels(qrels_path)
    qrels_sha = hashlib.sha256(qrels_path.read_bytes()).hexdigest()

    ans_queries = [q for q in dev_bench if q["is_answerable"]]
    llm_slice = select_dev_slice(dev_bench, per_emo=3)

    variants = {
        "fixed": chunk_fixed(docs, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP),
        "heading_aware": chunk_heading(docs),
        "hierarchical": chunk_hierarchical(docs),
    }

    results = {
        "split": "dev",
        "variable_changed": "chunking_strategy",
        "fixed_controls": {"retriever": "hybrid_rerank", "top_k": TOP_K, "model": BASE_MODEL, "qrels_sha256": qrels_sha},
        "definitions": {
            "context_groundedness": f"Proporsi dokumen relevan qrels di dalam Top-{TOP_K} konteks yang di-retrieve.",
            "answer_groundedness": "Proporsi output LLM yang seluruh evidence_spans muncul verbatim di teks ulasan dan seluruh policy_citations berasal dari dokumen yang di-retrieve.",
        },
        "variants": {},
    }

    for name, chunks in variants.items():
        eng = RetrievalEngine(chunks)
        runs = {}
        ctx_chars = []
        ctx_tokens_est = []
        ctx_rel_ratios = []

        for q in ans_queries:
            hits = eng.search_hybrid_rerank(q["text"], top_k=10, category=q.get("category"))
            runs[q["query_id"]] = [d["doc_id"] for d, _ in hits]

            top_hits = hits[:TOP_K]
            chars = sum(len(d["content"]) for d, _ in top_hits)
            ctx_chars.append(chars)
            ctx_tokens_est.append(round(chars / 4.0, 1))

            gold_docs = {did for did, rel in qrels.get(q["query_id"], {}).items() if rel > 0}
            top_ids = list(dict.fromkeys(d["doc_id"] for d, _ in top_hits))
            ctx_rel_ratios.append(len(set(top_ids) & gold_docs) / max(1, len(top_ids)))

        ir = eval_ranking(runs, qrels)
        index_bytes = len("\n".join(json.dumps(c, ensure_ascii=False) for c in chunks).encode("utf-8"))
        ans_ground = measure_llm_groundedness(eng, llm_slice)

        results["variants"][name] = {
            "index_size_chunks": len(chunks),
            "index_size_kb": round(index_bytes / 1024.0, 1),
            "avg_chunk_chars": round(float(np.mean([len(c["content"]) for c in chunks])), 1),
            "context_length_chars": round(float(np.mean(ctx_chars)), 1),
            "context_length_tokens_est": round(float(np.mean(ctx_tokens_est)), 1),
            "passage_recall@5": ir["recall@5"],
            "passage_recall@10": ir["recall@10"],
            "ndcg@5": ir["ndcg@5"],
            "ndcg@10": ir["ndcg@10"],
            "mrr@10": ir["mrr@10"],
            "context_groundedness": round(float(np.mean(ctx_rel_ratios)), 4),
            "answer_groundedness": ans_ground,
            "queries": ir["queries"],
            "llm_groundedness_samples": len(llm_slice),
        }

    best = max(results["variants"], key=lambda k: (results["variants"][k]["ndcg@10"], results["variants"][k]["answer_groundedness"]))
    results["selected"] = best

    out_path = RES_DIR / "ablation_a1.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps(results, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
