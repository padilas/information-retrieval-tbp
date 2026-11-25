import hashlib
import json
from config import BENCH_DIR, CHUNKING_STRATEGY, CORPUS_DIR, EMBED_MODEL, PROJECT_ROOT, RES_DIR, RRF_K
from ir_metrics import eval_ranking, load_jsonl, load_qrels
from retrieval import RetrievalEngine, apply_chunking, load_corpus

def main():
    RES_DIR.mkdir(parents=True, exist_ok=True)
    docs = apply_chunking(load_corpus(CORPUS_DIR / "corpus.jsonl"), CHUNKING_STRATEGY)
    dev_bench = load_jsonl(BENCH_DIR / "dev_benchmark_120.jsonl")
    qrels_path = BENCH_DIR / "qrels_dev.tsv"
    qrels = load_qrels(qrels_path)
    qrels_sha = hashlib.sha256(qrels_path.read_bytes()).hexdigest()

    ans_queries = [q for q in dev_bench if q["is_answerable"]]
    eng = RetrievalEngine(docs)

    methods = {
        "dense": lambda q: eng.search_dense(q["text"], top_k=10),
        "bm25": lambda q: eng.search_bm25(q["text"], top_k=10),
        "hybrid": lambda q: eng.search_hybrid(q["text"], top_k=10),
        "hybrid_rerank": lambda q: eng.search_hybrid_rerank(q["text"], top_k=10, category=q.get("category")),
    }

    results = {
        "split": "dev",
        "variable_changed": "retrieval_method",
        "fixed_controls": {
            "chunking": CHUNKING_STRATEGY,
            "embed_model": EMBED_MODEL,
            "rrf_k": RRF_K,
            "qrels_file": qrels_path.relative_to(PROJECT_ROOT).as_posix(),
            "qrels_sha256": qrels_sha,
        },
        "methods": {},
    }

    for name, fn in methods.items():
        runs = {q["query_id"]: [d["doc_id"] for d, _ in fn(q)] for q in ans_queries}
        results["methods"][name] = eval_ranking(runs, qrels)

    best = max(results["methods"], key=lambda k: results["methods"][k]["ndcg@10"])
    results["selected"] = best

    out_path = RES_DIR / "ablation_a2.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps(results, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
