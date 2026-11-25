import hashlib
import json
import math
import pickle
import re
from collections import Counter
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from config import (
    BM25_B,
    BM25_K1,
    CAT_BOOST,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBED_MODEL,
    POLICY_BOOST,
    PROC_DIR,
    RRF_K,
    SEED,
)

STOPWORDS = {
    "yang", "di", "dan", "ini", "dari", "untuk", "pada", "adalah", "ke", "itu",
    "dengan", "karena", "oleh", "saat", "atau", "juga", "akan", "ada", "saya",
    "kami", "anda", "dia", "mereka", "kita", "ia", "pun", "nya", "ya", "sih", "aja",
}

KEEP_NYA = {"hanya", "punya", "tanya", "nyata", "nyaman", "nyampe", "nyesel"}
NEGATORS = {"tidak", "kurang", "belum", "jangan", "bukan", "tanpa"}

SLANG_MAP = {
    "yg": "yang", "ga": "tidak", "gak": "tidak", "gk": "tidak", "tdk": "tidak",
    "ngga": "tidak", "nggak": "tidak", "kagak": "tidak", "nda": "tidak", "ndak": "tidak",
    "bgt": "banget", "bgtt": "banget", "bangettt": "banget", "sekali": "banget",
    "bgs": "bagus", "bagusss": "bagus", "mantab": "mantap", "mantap": "mantap",
    "mantul": "mantap banget", "jos": "mantap", "joss": "mantap", "ajib": "mantap",
    "brg": "barang", "brng": "barang", "produk": "barang",
    "udh": "sudah", "sdh": "sudah", "uda": "sudah", "udah": "sudah",
    "blm": "belum", "blom": "belum", "dgn": "dengan", "dg": "dengan",
    "utk": "untuk", "buat": "untuk", "tp": "tetapi", "tpi": "tetapi", "tapi": "tetapi",
    "jg": "juga", "jga": "juga", "klo": "kalau", "kl": "kalau", "kalo": "kalau",
    "dr": "dari", "sm": "sama", "ama": "sama", "krn": "karena", "karna": "karena",
    "sy": "saya", "aku": "saya", "gw": "saya", "gue": "saya",
    "seler": "seller", "penjual": "seller", "pelapak": "seller", "toko": "seller",
    "recsel": "recommended seller", "recomended": "recommended", "rekomen": "recommended",
    "cepet": "cepat", "cpt": "cepat", "sampe": "sampai", "nyampe": "sampai",
    "pake": "pakai", "dipake": "pakai", "dipakai": "pakai",
    "pengiriman": "kirim", "dikirim": "kirim", "pengemasan": "packing", "packaging": "packing",
    "kemasan": "packing", "dikemas": "packing", "berfungsi": "fungsi", "berjalan": "fungsi",
    "mengecewakan": "kecewa", "nyesel": "sedih", "menyesal": "sedih", "zonk": "kecewa rusak",
    "pesanan": "pesan", "order": "pesan", "ori": "original", "asli": "original",
    "ok": "oke", "okey": "oke", "smoga": "semoga", "bener": "benar",
    "trimakasih": "terima kasih", "trims": "terima kasih", "makasi": "terima kasih",
    "makasih": "terima kasih", "thx": "terima kasih", "tq": "terima kasih", "thanks": "terima kasih",
}

_MODEL_CACHE = {}
_QUERY_VEC_CACHE = {}
_RERANK_BUNDLE_CACHE = {}
EMB_CACHE_DIR = PROC_DIR / "emb_cache"

def _get_embedder(model_name=EMBED_MODEL):
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]

def _strip_suffix(word):
    if len(word) > 5 and word not in KEEP_NYA:
        for suf in ("nya", "lah", "pun"):
            if word.endswith(suf):
                return word[:-len(suf)]
    return word

def clean_natural(text):
    s = str(text).replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", s).strip()

def normalize_text(text):
    s = str(text).lower()
    s = re.sub(r"(.)\1{2,}", r"\1", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    expanded = []
    for raw in s.split():
        base = _strip_suffix(raw)
        mapped = SLANG_MAP.get(base, SLANG_MAP.get(raw, base))
        expanded.extend(mapped.split())
    return " ".join(expanded)

def tokenize(text, include_bigrams=False):
    words = normalize_text(text).split()
    unigrams = []
    i = 0
    while i < len(words):
        w = words[i]
        if w in NEGATORS and i + 1 < len(words):
            nxt = words[i + 1]
            if nxt not in STOPWORDS:
                unigrams.append(f"{w}_{nxt}")
                i += 2
                continue
        if w not in STOPWORDS and len(w) > 1:
            unigrams.append(w)
        i += 1
    if not include_bigrams or len(unigrams) < 2:
        return unigrams
    bigrams = [f"{unigrams[j]}__{unigrams[j+1]}" for j in range(len(unigrams) - 1)]
    return unigrams + bigrams

def load_corpus(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def chunk_fixed(docs, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    step = max(20, size - overlap)
    for d in docs:
        text = d.get("content", "")
        if len(text) <= size:
            chunks.append({**d, "chunk_id": f"{d['doc_id']}_c0", "content": text})
            continue
        start, idx = 0, 0
        while start < len(text):
            end = min(len(text), start + size)
            chunks.append({**d, "chunk_id": f"{d['doc_id']}_c{idx}", "content": text[start:end]})
            if end >= len(text):
                break
            start += step
            idx += 1
    return chunks

def chunk_heading(docs):
    chunks = []
    for d in docs:
        text = d.get("content", "")
        parts = [p.strip() for p in re.split(r"(?=\n#+ )", text) if p.strip()]
        if not parts:
            parts = [text]
        for idx, p in enumerate(parts):
            chunks.append({**d, "chunk_id": f"{d['doc_id']}_h{idx}", "content": p})
    return chunks

def chunk_hierarchical(docs):
    chunks = []
    for d in docs:
        text = d.get("content", "")
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        parent = lines[0] if lines else text[:120]
        children = lines[1:] if len(lines) > 1 else [text]
        for idx, ch in enumerate(children):
            merged = f"{parent} | {ch}" if ch != parent else ch
            chunks.append({**d, "chunk_id": f"{d['doc_id']}_p{idx}", "parent": parent, "content": merged})
    return chunks

def apply_chunking(docs, strategy="heading_aware", size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if strategy == "fixed":
        return chunk_fixed(docs, size=size, overlap=overlap)
    if strategy == "heading_aware":
        return chunk_heading(docs)
    if strategy == "hierarchical":
        return chunk_hierarchical(docs)
    return docs

class BM25:
    def __init__(self, texts, k1=BM25_K1, b=BM25_B):
        self.k1 = k1
        self.b = b
        self.n = len(texts)
        self.doc_tokens = [Counter(tokenize(t, include_bigrams=True)) for t in texts]
        self.doc_lens = [sum(c.values()) for c in self.doc_tokens]
        self.avg_len = sum(self.doc_lens) / max(1, self.n)
        df = Counter()
        for c in self.doc_tokens:
            for tok in c:
                df[tok] += 1
        self.idf = {tok: math.log(1 + (self.n - freq + 0.5) / (freq + 0.5)) for tok, freq in df.items()}

    def scores(self, query):
        q_toks = tokenize(query, include_bigrams=True)
        out = np.zeros(self.n, dtype=float)
        for tok in q_toks:
            idf = self.idf.get(tok)
            if idf is None:
                continue
            for i, counts in enumerate(self.doc_tokens):
                tf = counts.get(tok, 0)
                if tf == 0:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * (self.doc_lens[i] / max(1e-6, self.avg_len)))
                out[i] += idf * (tf * (self.k1 + 1)) / denom
        return out

def _encode_cached(texts, prefix, model_name=EMBED_MODEL):
    EMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.md5((model_name + "|" + prefix + "|" + "\n".join(texts)).encode("utf-8")).hexdigest()
    cache_file = EMB_CACHE_DIR / f"{digest}.npy"
    if cache_file.exists():
        return np.load(cache_file)
    model = _get_embedder(model_name)
    prefixed = [f"{prefix}: {t}" for t in texts]
    vecs = model.encode(prefixed, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    vecs = np.asarray(vecs, dtype=np.float32)
    np.save(cache_file, vecs)
    return vecs

def _get_train_rerank_bundle(model_name=EMBED_MODEL):
    train_path = PROC_DIR / "train.csv"
    if not train_path.exists():
        return None
    train_hash = hashlib.md5(train_path.read_bytes()).hexdigest()[:12]
    safe_model = re.sub(r"[^a-zA-Z0-9_-]", "_", model_name)
    key = f"bundle_v2_{safe_model}_{train_hash}"
    if key in _RERANK_BUNDLE_CACHE:
        return _RERANK_BUNDLE_CACHE[key]

    EMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    bundle_path = EMB_CACHE_DIR / f"{key}.pkl"
    if bundle_path.exists():
        with open(bundle_path, "rb") as f:
            bundle = pickle.load(f)
        _RERANK_BUNDLE_CACHE[key] = bundle
        return bundle

    # dilatih 100% di train.csv: gabungan E5 + Word/Char TF-IDF untuk afinitas sentimen & emosi di reranker
    df = pd.read_csv(train_path)
    nat_texts = [clean_natural(t) for t in df["Customer Review"]]
    norm_texts = [" ".join(tokenize(t, include_bigrams=True)) for t in df["Customer Review"]]

    e5_mat = csr_matrix(_encode_cached(nat_texts, prefix="query", model_name=model_name))
    w_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    c_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True)
    X_train = hstack([e5_mat * 1.5, w_vec.fit_transform(norm_texts), c_vec.fit_transform([t.lower() for t in nat_texts])])

    sent_clf = LogisticRegression(C=2.5, max_iter=600, class_weight="balanced", random_state=SEED)
    sent_clf.fit(X_train, df["Sentiment"].astype(str).tolist())

    emo_clf = LogisticRegression(C=2.0, max_iter=600, class_weight="balanced", random_state=SEED)
    emo_clf.fit(X_train, df["Emotion"].astype(str).tolist())

    bundle = {"w_vec": w_vec, "c_vec": c_vec, "sent_clf": sent_clf, "emo_clf": emo_clf}
    with open(bundle_path, "wb") as f:
        pickle.dump(bundle, f)
    _RERANK_BUNDLE_CACHE[key] = bundle
    return bundle

class RetrievalEngine:
    def __init__(
        self,
        docs,
        embed_model=EMBED_MODEL,
        k1=BM25_K1,
        b=BM25_B,
        rrf_k=RRF_K,
        cat_boost=CAT_BOOST,
        policy_boost=POLICY_BOOST,
    ):
        self.docs = docs
        self.embed_model = embed_model
        self.rrf_k = rrf_k
        self.cat_boost = cat_boost
        self.policy_boost = policy_boost

        bm25_texts = [f"{d.get('title', '')} {d.get('content', '')}".strip() for d in docs]
        dense_texts = [
            clean_natural(f"{d.get('title', '')}: {d.get('content', '')}" if d.get("doc_type") == "policy" else d.get("content", ""))
            for d in docs
        ]
        self.bm25 = BM25(bm25_texts, k1=k1, b=b)
        self.dense_mat = _encode_cached(dense_texts, prefix="passage", model_name=embed_model)
        self.char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.char_mat = self.char_vec.fit_transform([t.lower() for t in dense_texts])
        self.rerank_bundle = _get_train_rerank_bundle(model_name=embed_model)

    def _pass_filter(self, doc, flt):
        if not flt:
            return True
        return all(doc.get(k) == v for k, v in flt.items())

    def _dedup_top_docs(self, scored_indices, scores, top_k, flt=None):
        best_by_doc = {}
        for i in scored_indices:
            d = self.docs[i]
            if not self._pass_filter(d, flt):
                continue
            did = d["doc_id"]
            sc = float(scores[i])
            if did not in best_by_doc or sc > best_by_doc[did][1]:
                best_by_doc[did] = (d, sc)
            if len(best_by_doc) >= top_k:
                break
        res = list(best_by_doc.values())
        res.sort(key=lambda x: x[1], reverse=True)
        return res[:top_k]

    def _query_vec(self, query):
        nat_q = clean_natural(query)
        key = (self.embed_model, nat_q)
        if key not in _QUERY_VEC_CACHE:
            model = _get_embedder(self.embed_model)
            _QUERY_VEC_CACHE[key] = np.asarray(
                model.encode([f"query: {nat_q}"], normalize_embeddings=True, show_progress_bar=False)[0],
                dtype=np.float32,
            )
        return _QUERY_VEC_CACHE[key]

    def _dense_scores(self, query):
        return self.dense_mat @ self._query_vec(query)

    def search_bm25(self, query, top_k=10, flt=None):
        sc = self.bm25.scores(query)
        order = np.argsort(sc)[::-1]
        return self._dedup_top_docs(order, sc, top_k, flt=flt)

    def search_dense(self, query, top_k=10, flt=None):
        sims = self._dense_scores(query)
        order = np.argsort(sims)[::-1]
        return self._dedup_top_docs(order, sims, top_k, flt=flt)

    def search_hybrid(self, query, top_k=10, rrf_k=None, flt=None):
        rk = rrf_k if rrf_k is not None else self.rrf_k
        pool_k = max(60, top_k * 8)
        b_res = self.search_bm25(query, top_k=pool_k, flt=flt)
        d_res = self.search_dense(query, top_k=pool_k, flt=flt)

        ranks = {}
        for r, (d, _) in enumerate(b_res, 1):
            ranks.setdefault(d["doc_id"], {"doc": d})["b"] = r
        for r, (d, _) in enumerate(d_res, 1):
            ranks.setdefault(d["doc_id"], {"doc": d})["d"] = r

        scored = []
        for v in ranks.values():
            score = 0.45 / (rk + v.get("b", 1000)) + 0.55 / (rk + v.get("d", 1000))
            scored.append((v["doc"], score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def _predict_query_affect(self, query, q_vec):
        if self.rerank_bundle is None:
            return {}, {}
        nat_q = clean_natural(query)
        norm_q = " ".join(tokenize(query, include_bigrams=True))
        w_vec = self.rerank_bundle["w_vec"]
        c_vec = self.rerank_bundle["c_vec"]
        Xq = hstack([csr_matrix(q_vec.reshape(1, -1)) * 1.5, w_vec.transform([norm_q]), c_vec.transform([nat_q.lower()])])
        s_clf = self.rerank_bundle["sent_clf"]
        e_clf = self.rerank_bundle["emo_clf"]
        s_probs = {c: float(p) for c, p in zip(s_clf.classes_, s_clf.predict_proba(Xq)[0])}
        e_probs = {c: float(p) for c, p in zip(e_clf.classes_, e_clf.predict_proba(Xq)[0])}
        return s_probs, e_probs

    def search_hybrid_rerank(self, query, top_k=10, category=None, version=None):
        flt = {"version": version} if version else None
        q_vec = self._query_vec(query)
        dense_all = self.dense_mat @ q_vec
        char_all = cosine_similarity(self.char_vec.transform([clean_natural(query).lower()]), self.char_mat).ravel()
        bm25_all = self.bm25.scores(query)
        max_bm25 = max(1e-6, float(np.max(bm25_all)))

        sent_probs, emo_probs = self._predict_query_affect(query, q_vec)

        cands = self.search_hybrid(query, top_k=120, flt=flt)
        cand_map = {d["doc_id"]: rrf_sc for d, rrf_sc in cands}

        doc_best = {}
        for i, d in enumerate(self.docs):
            if not self._pass_filter(d, flt):
                continue
            did = d["doc_id"]
            ds = float(dense_all[i])
            cs = float(char_all[i])
            bs = float(bm25_all[i]) / max_bm25
            comb = 0.55 * ds + 0.30 * cs + 0.15 * bs
            if did not in doc_best or comb > doc_best[did]["comb"]:
                doc_best[did] = {"doc": d, "dense": ds, "char": cs, "bm25": bs, "comb": comb}

        # sertakan seluruh dokumen policy dan top exemplar dari prediksi emosi teratas agar recall tidak terpotong
        top_emos = sorted(emo_probs, key=emo_probs.get, reverse=True)[:2]
        for did, info in doc_best.items():
            d = info["doc"]
            if d.get("doc_type") == "policy" or d.get("emotion") in top_emos:
                cand_map.setdefault(did, 0.0)

        q_toks = set(tokenize(query, include_bigrams=True))
        rrf_norm = float(self.rrf_k) + 1.0

        pol_docs = [
            did for did, info in doc_best.items()
            if info["doc"].get("doc_type") == "policy"
            and info["doc"].get("version", "v1.0") == (version or "v1.0")
            and info["doc"].get("emotion") in emo_probs
        ]
        top_pol = None
        if pol_docs:
            top_pol = max(
                pol_docs,
                key=lambda did: 0.70 * emo_probs.get(doc_best[did]["doc"].get("emotion"), 0.0) + 0.30 * doc_best[did]["comb"],
            )

        out = []
        for did, rrf_sc in cand_map.items():
            info = doc_best[did]
            d = info["doc"]
            dt = set(tokenize(d.get("content", ""), include_bigrams=True))
            lex = len(q_toks & dt) / max(1, len(q_toks | dt))

            d_emo = d.get("emotion")
            d_sent = d.get("sentiment")
            e_aff = emo_probs.get(d_emo, 0.0) if d_emo else 0.0
            s_aff = sent_probs.get(d_sent, 0.0) if d_sent else 0.0

            sim_part = 0.45 * max(0.0, info["dense"]) + 0.25 * info["char"] + 0.15 * lex + 0.15 * (rrf_sc * rrf_norm)
            aff_part = 0.70 * e_aff + 0.30 * s_aff
            score = sim_part * (0.25 + 0.75 * aff_part)
            if did == top_pol:
                score = (0.45 + 0.55 * aff_part) * self.policy_boost
            elif category and d.get("category") == category:
                score *= self.cat_boost
            out.append((d, score))

        out.sort(key=lambda x: x[1], reverse=True)
        return out[:top_k]
