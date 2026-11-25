import json
from config import BENCH_DIR, CORPUS_DIR, RES_DIR
from ir_metrics import eval_ranking, load_jsonl, load_qrels
from retrieval import RetrievalEngine, chunk_fixed, chunk_heading, load_corpus

def eval_engine(eng, queries, qrels, mode="hybrid_rerank"):
    runs = {}
    for q in queries:
        if mode == "bm25":
            hits = eng.search_bm25(q["text"], top_k=10)
        else:
            hits = eng.search_hybrid_rerank(q["text"], top_k=10, category=q.get("category"))
        runs[q["query_id"]] = [d["doc_id"] for d, _ in hits]
    return eval_ranking(runs, qrels)

def main():
    RES_DIR.mkdir(parents=True, exist_ok=True)
    raw_docs = load_corpus(CORPUS_DIR / "corpus.jsonl")
    dev_bench = load_jsonl(BENCH_DIR / "dev_benchmark_120.jsonl")
    qrels = load_qrels(BENCH_DIR / "qrels_dev.tsv")

    ans_queries = [q for q in dev_bench if q["is_answerable"]]
    out = {"split": "dev", "queries": len(ans_queries), "sweeps": {}}

    # 1. Sliding window chunk_size x overlap
    win_grid = [(120, 30), (160, 40), (220, 55), (280, 70), (360, 90)]
    win_res = {}
    for sz, ov in win_grid:
        chunks = chunk_fixed(raw_docs, size=sz, overlap=ov)
        eng = RetrievalEngine(chunks)
        win_res[f"size_{sz}_ov_{ov}"] = {"chunks": len(chunks), **eval_engine(eng, ans_queries, qrels)}
    best_win = max(win_res, key=lambda k: win_res[k]["ndcg@10"])
    out["sweeps"]["sliding_window"] = {"results": win_res, "best": best_win}

    # 2. BM25 (k1, b) pada chunk heading_aware
    h_chunks = chunk_heading(raw_docs)
    bm25_grid = [(1.2, 0.50), (1.2, 0.75), (1.5, 0.75), (1.5, 0.90), (1.8, 0.85)]
    bm25_res = {}
    for k1, b in bm25_grid:
        eng = RetrievalEngine(h_chunks, k1=k1, b=b)
        bm25_res[f"k1_{k1}_b_{b}"] = eval_engine(eng, ans_queries, qrels, mode="bm25")
    best_bm25 = max(bm25_res, key=lambda k: bm25_res[k]["ndcg@10"])
    best_k1, best_b = [float(x) for x in [best_bm25.split("_")[1], best_bm25.split("_")[3]]]
    out["sweeps"]["bm25_params"] = {"results": bm25_res, "best": best_bm25}

    # 3. RRF constant rrf_k
    rrf_res = {}
    for rk in [10, 20, 40, 60]:
        eng = RetrievalEngine(h_chunks, k1=best_k1, b=best_b, rrf_k=rk)
        rrf_res[f"rrf_{rk}"] = eval_engine(eng, ans_queries, qrels)
    best_rrf_key = max(rrf_res, key=lambda k: rrf_res[k]["ndcg@10"])
    best_rk = int(best_rrf_key.split("_")[1])
    out["sweeps"]["rrf_k"] = {"results": rrf_res, "best": best_rrf_key}

    # 4. Reranker category & policy boost
    boost_grid = [(1.10, 1.25), (1.15, 1.50), (1.20, 1.80), (1.25, 2.10), (1.30, 2.40)]
    boost_res = {}
    for cb, pb in boost_grid:
        eng = RetrievalEngine(h_chunks, k1=best_k1, b=best_b, rrf_k=best_rk, cat_boost=cb, policy_boost=pb)
        boost_res[f"cat_{cb}_pol_{pb}"] = eval_engine(eng, ans_queries, qrels)
    best_boost = max(boost_res, key=lambda k: boost_res[k]["ndcg@10"])
    out["sweeps"]["reranker_boosts"] = {"results": boost_res, "best": best_boost}

    out_path = RES_DIR / "tuning_dev.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
