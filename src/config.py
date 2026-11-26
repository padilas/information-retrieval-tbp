import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
ROOT = Path(__file__).resolve().parent.parent

def _load_env():
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

_load_env()

_proj_env = os.getenv("PROJECT_ROOT", "").strip()
PROJECT_ROOT = Path(_proj_env) if _proj_env else ROOT
RAW_CSV = PROJECT_ROOT / os.getenv("RAW_CSV", "data/PRDECT-ID Dataset.csv")
PROC_DIR = PROJECT_ROOT / "data" / "processed"
CORPUS_DIR = PROJECT_ROOT / "data" / "corpus"
BENCH_DIR = PROJECT_ROOT / "data" / "benchmark"
RES_DIR = PROJECT_ROOT / "results"

BASE_MODEL = os.getenv("BASE_MODEL", "qwen2.5:1.5b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
SEED = int(os.getenv("SEED", "42"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "300"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "45"))

INDEX_VERSION = os.getenv("INDEX_VERSION", "corpus_v3")
CHUNKING_STRATEGY = os.getenv("CHUNKING_STRATEGY", "heading_aware")
TOP_K = int(os.getenv("TOP_K", "3"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "240"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "60"))
BM25_K1 = float(os.getenv("BM25_K1", "1.2"))
BM25_B = float(os.getenv("BM25_B", "0.4"))
RRF_K = int(os.getenv("RRF_K", "30"))
CAT_BOOST = float(os.getenv("CAT_BOOST", "1.15"))
POLICY_BOOST = float(os.getenv("POLICY_BOOST", "1.25"))
MAX_CTX_CHARS = int(os.getenv("MAX_CTX_CHARS", "1500"))
EXEMPLARS_PER_CLASS = int(os.getenv("EXEMPLARS_PER_CLASS", "120"))

A7_LOW_TOPK = int(os.getenv("A7_LOW_TOPK", "2"))
A7_LOW_CHARS = int(os.getenv("A7_LOW_CHARS", "900"))
A7_HIGH_TOPK = int(os.getenv("A7_HIGH_TOPK", "5"))
A7_HIGH_CHARS = int(os.getenv("A7_HIGH_CHARS", "2800"))
A7_SAMPLES = int(os.getenv("A7_SAMPLES", "20"))
BASELINES_DEV_SAMPLES = int(os.getenv("BASELINES_DEV_SAMPLES", "24"))

B4_SUFFICIENCY_THRESHOLD = float(os.getenv("B4_SUFFICIENCY_THRESHOLD", "0.52"))
B4_MAX_RETRIES = int(os.getenv("B4_MAX_RETRIES", "2"))
B5_LORA_ADAPTER_DIR = PROJECT_ROOT / os.getenv("B5_LORA_ADAPTER_DIR", "models/qwen25_15b_prdect_lora")
B5_LORA_MODEL = os.getenv("B5_LORA_MODEL", BASE_MODEL)
B6_MAX_TOOL_CALLS = int(os.getenv("B6_MAX_TOOL_CALLS", "4"))
B6_TOKEN_BUDGET = int(os.getenv("B6_TOKEN_BUDGET", "1200"))
