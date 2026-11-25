import csv
import json
import math
import numpy as np

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def load_qrels(path):
    qrels = {}
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            qrels.setdefault(row["query_id"], {})[row["doc_id"]] = int(row["relevance"])
    return qrels

def eval_ranking(runs, qrels):
    r5, r10, ndcg5, ndcg10, mrr10 = [], [], [], [], []
    for qid, ranked_ids in runs.items():
        gold = qrels.get(qid)
        if not gold:
            continue
        rel_set = {d for d, r in gold.items() if r > 0}
        if not rel_set:
            continue

        seen = []
        for did in ranked_ids:
            if did not in seen:
                seen.append(did)

        r5.append(len(set(seen[:5]) & rel_set) / len(rel_set))
        r10.append(len(set(seen[:10]) & rel_set) / len(rel_set))

        rr = 0.0
        for rank, did in enumerate(seen[:10], 1):
            if did in rel_set:
                rr = 1.0 / rank
                break
        mrr10.append(rr)

        dcg5 = sum((2 ** gold.get(did, 0) - 1) / math.log2(rank + 1) for rank, did in enumerate(seen[:5], 1) if gold.get(did, 0) > 0)
        ideal5 = sorted(gold.values(), reverse=True)[:5]
        idcg5 = sum((2 ** rel - 1) / math.log2(rank + 1) for rank, rel in enumerate(ideal5, 1))
        ndcg5.append(dcg5 / idcg5 if idcg5 > 0 else 0.0)

        dcg10 = sum((2 ** gold.get(did, 0) - 1) / math.log2(rank + 1) for rank, did in enumerate(seen[:10], 1) if gold.get(did, 0) > 0)
        ideal10 = sorted(gold.values(), reverse=True)[:10]
        idcg10 = sum((2 ** rel - 1) / math.log2(rank + 1) for rank, rel in enumerate(ideal10, 1))
        ndcg10.append(dcg10 / idcg10 if idcg10 > 0 else 0.0)

    return {
        "recall@5": round(float(np.mean(r5)), 4),
        "recall@10": round(float(np.mean(r10)), 4),
        "ndcg@5": round(float(np.mean(ndcg5)), 4),
        "ndcg@10": round(float(np.mean(ndcg10)), 4),
        "mrr@10": round(float(np.mean(mrr10)), 4),
        "queries": len(ndcg10),
    }
