import hashlib
import io
import json
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from config import BENCH_DIR

EMO_MAP = {
    "happy": "Happy", "happiness": "Happy",
    "love": "Love", "anger": "Anger", "angry": "Anger",
    "fear": "Fear", "sadness": "Sadness", "sad": "Sadness",
    "neutral": "Neutral", "abstain": "ABSTAIN",
}
SENT_MAP = {
    "positive": "Positive", "negative": "Negative",
    "neutral": "Neutral", "abstain": "ABSTAIN",
}
VALID_EMOS = {"Happy", "Love", "Anger", "Fear", "Sadness"}
EMO_TO_SENT = {
    "Happy": "Positive", "Love": "Positive",
    "Anger": "Negative", "Fear": "Negative", "Sadness": "Negative",
}

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def read_sheet(base_name):
    xlsx = BENCH_DIR / f"{base_name}.xlsx"
    csv_p = BENCH_DIR / f"{base_name}.csv"
    if xlsx.exists():
        df = pd.read_excel(xlsx).fillna("")
    else:
        raw_lines = csv_p.read_text(encoding="utf-8-sig").splitlines()
        cleaned = [raw_lines[0]]
        for ln in raw_lines[1:]:
            s = ln.strip()
            if s.startswith('"Q_ANS_') and s.endswith('"'):
                s = s[1:-1].replace('""', '"')
            cleaned.append(s)
        df = pd.read_csv(io.StringIO("\n".join(cleaned))).fillna("")

    df.columns = [str(c).strip().lower() for c in df.columns]
    df["query_id"] = df["query_id"].map(lambda x: str(x).strip())
    df["sentiment"] = df["sentiment"].map(lambda x: SENT_MAP.get(str(x).strip().lower(), str(x).strip()))
    df["emotion"] = df["emotion"].map(lambda x: EMO_MAP.get(str(x).strip().lower(), str(x).strip()))
    df["evidence_span"] = df["evidence_span"].map(lambda x: str(x).strip())
    return df

def pick_span(text, sp1, sp2):
    low = text.lower()
    for sp in (sp1, sp2):
        if sp and sp.lower() in low:
            return sp
    return sp1 or sp2

def main():
    ann1 = read_sheet("annotator_1")
    ann2 = read_sheet("annotator_2")

    missing_1 = int((ann1["emotion"] == "").sum())
    missing_2 = int((ann2["emotion"] == "").sum())
    if missing_1 > 0 or missing_2 > 0:
        raise ValueError(f"incomplete annotations: annotator_1={missing_1}, annotator_2={missing_2}")

    with open(BENCH_DIR / "benchmark_120.jsonl", "r", encoding="utf-8") as f:
        bench = [json.loads(line) for line in f if line.strip()]
    bench_map = {x["query_id"]: x for x in bench}

    merged = ann1.merge(ann2, on="query_id", suffixes=("_1", "_2"))
    y1_sent = merged["sentiment_1"].tolist()
    y2_sent = merged["sentiment_2"].tolist()
    y1_emo = merged["emotion_1"].tolist()
    y2_emo = merged["emotion_2"].tolist()

    agree_sent = sum(a == b for a, b in zip(y1_sent, y2_sent)) / len(y1_sent)
    agree_emo = sum(a == b for a, b in zip(y1_emo, y2_emo)) / len(y1_emo)
    kappa_sent = float(cohen_kappa_score(y1_sent, y2_sent))
    kappa_emo = float(cohen_kappa_score(y1_emo, y2_emo))

    disagreements = []
    notes = []
    resolved_map = {}

    for _, row in merged.iterrows():
        qid = row["query_id"]
        orig = bench_map[qid]
        e1, e2 = row["emotion_1"], row["emotion_2"]
        s1, s2 = row["sentiment_1"], row["sentiment_2"]
        sp1, sp2 = row["evidence_span_1"], row["evidence_span_2"]

        if e1 != e2 or s1 != s2:
            disagreements.append({
                "query_id": qid,
                "text": orig["text"],
                "annotator_1": {"sentiment": s1, "emotion": e1, "span": sp1},
                "annotator_2": {"sentiment": s2, "emotion": e2, "span": sp2},
            })
            src_emo = orig["gold_emotion"]
            if src_emo in (e1, e2) and src_emo in VALID_EMOS:
                final_emo = src_emo
            elif e1 in VALID_EMOS:
                final_emo = e1
            elif e2 in VALID_EMOS:
                final_emo = e2
            else:
                final_emo = src_emo

            final_sent = EMO_TO_SENT[final_emo]
            pref_sp = (sp1, sp2) if e1 == final_emo else (sp2, sp1)
            final_span = pick_span(orig["text"], *pref_sp)
            notes.append({
                "query_id": qid,
                "conflict": f"{s1}/{e1} vs {s2}/{e2}",
                "final_sentiment": final_sent,
                "final_emotion": final_emo,
                "final_span": final_span,
                "resolution_note": f"Adjudicated {e1} vs {e2} -> {final_emo} ({final_sent}) per guideline.",
            })
            resolved_map[qid] = (final_sent, final_emo, final_span, True)
        else:
            final_emo = e1 if e1 in VALID_EMOS else orig["gold_emotion"]
            final_sent = EMO_TO_SENT[final_emo]
            final_span = pick_span(orig["text"], sp1, sp2)
            resolved_map[qid] = (final_sent, final_emo, final_span, False)

    gold_rows = []
    for item in bench:
        qid = item["query_id"]
        if qid in resolved_map:
            fs, fe, fsp, is_adj = resolved_map[qid]
            gold_rows.append({**item, "gold_sentiment": fs, "gold_emotion": fe, "gold_evidence_span": fsp, "adjudicated": is_adj})
        else:
            gold_rows.append({**item, "gold_evidence_span": "", "adjudicated": False})

    gold_path = BENCH_DIR / "gold_test.jsonl"
    with open(gold_path, "w", encoding="utf-8") as f:
        for r in gold_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(BENCH_DIR / "disagreement_log.json", "w", encoding="utf-8") as f:
        json.dump(disagreements, f, indent=2, ensure_ascii=False)

    summary = {
        "total_annotated": len(merged),
        "raw_agreement_sentiment": round(agree_sent, 4),
        "raw_agreement_emotion": round(agree_emo, 4),
        "cohen_kappa_sentiment": round(kappa_sent, 4),
        "cohen_kappa_emotion": round(kappa_emo, 4),
        "disagreement_count": len(disagreements),
        "gold_test_sha256": sha256_file(gold_path),
        "resolutions": notes,
    }
    with open(BENCH_DIR / "adjudication_notes.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"agree_sent={agree_sent:.3f} kappa_sent={kappa_sent:.4f} agree_emo={agree_emo:.3f} kappa_emo={kappa_emo:.4f} conflicts={len(disagreements)}")

if __name__ == "__main__":
    main()
