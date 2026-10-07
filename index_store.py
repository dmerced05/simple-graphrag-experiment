# step 5: save everything the query side needs to INDEX_DIR, and load it back
import json
import sys

import numpy as np

import config
from knowledge_graph import graph_from_json, graph_to_json


def save_index(graph, chunks, entity_keys, entity_vectors, chunk_vectors):
    index_dir = config.INDEX_DIR
    # remember how this index was built, so benchmark results can say which chunking they used
    method = chunks[0]["method"] if chunks else config.CHUNK_METHOD
    meta = {"chunk_method": method, "chunk_chars": config.CHUNK_CHARS,
            "chunk_overlap": config.CHUNK_OVERLAP if method == "fixed" else None,  # only the fixed method overlaps
            "chunk_count": len(chunks), "chat_model": config.CHAT_MODEL, "embed_model": config.EMBED_MODEL}
    (index_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    (index_dir / "graph.json").write_text(json.dumps(graph_to_json(graph), indent=1))
    (index_dir / "chunks.json").write_text(json.dumps(chunks, indent=1))
    (index_dir / "entity_keys.json").write_text(json.dumps(entity_keys))
    np.save(index_dir / "entity_vectors.npy", entity_vectors)
    np.save(index_dir / "chunk_vectors.npy", chunk_vectors)


def load_meta():
    # indexes built before meta.json existed used the original paragraph chunking
    path = config.INDEX_DIR / "meta.json"
    return json.loads(path.read_text()) if path.exists() else {"chunk_method": "paragraph"}


def load_index():
    index_dir = config.INDEX_DIR
    if not (index_dir / "graph.json").exists():
        sys.exit("No index yet. Run: python graphrag.py index")
    graph = graph_from_json(json.loads((index_dir / "graph.json").read_text()))
    chunks = json.loads((index_dir / "chunks.json").read_text())
    entity_keys = json.loads((index_dir / "entity_keys.json").read_text())
    entity_vectors = np.load(index_dir / "entity_vectors.npy")
    chunk_vectors = np.load(index_dir / "chunk_vectors.npy")
    return graph, chunks, entity_keys, entity_vectors, chunk_vectors
