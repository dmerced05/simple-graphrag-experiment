# settings shared by every step; all of them can be overridden with environment variables
import os
from pathlib import Path

CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen2.5:7b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", "index"))
WIKI_DIR = Path(os.getenv("WIKI_DIR", "wiki"))
KEEP_ALIVE = os.getenv("KEEP_ALIVE", "30m")  # keep models in memory between calls so reloads don't pollute timings
CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", "200"))  # small so the sample docs split up; raise to ~1000 for real docs
CHUNK_METHOD = os.getenv("CHUNK_METHOD", "paragraph")  # paragraph | fixed | structure (see chunking.py)
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))  # fixed method only: characters repeated between neighbors
