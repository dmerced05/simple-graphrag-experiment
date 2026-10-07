# steps 4 and 5: embedding.py and index_store.py
import json

import numpy as np
import pytest

import config
from embedding import embed, entity_text, ns_to_ms
from index_store import load_index, load_meta


def test_vectors_are_unit_length(fake_ollama):
    vectors = embed(["one text", "another text", "a third"])
    assert vectors.shape[0] == 3
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


def test_embed_batches(fake_ollama):
    embed([f"text {n}" for n in range(5)], batch_size=2)
    assert [len(batch) for batch in fake_ollama.embed_calls] == [2, 2, 1]


def test_embed_reports_load_time(fake_ollama):
    fake_ollama.embed_load_ns = 2_000_000
    stats = {}
    embed(["a", "b", "c"], batch_size=2, stats=stats)
    assert stats["load_ms"] == pytest.approx(4.0)


def test_ns_to_ms():
    assert ns_to_ms(1_500_000) == 1.5
    assert ns_to_ms(None) == 0.0


def test_entity_text():
    assert entity_text({"name": "Redwood", "type": "product", "descriptions": ["a cache", "leaky"]}) == \
        "Redwood (product): a cache leaky"


def test_index_round_trip(built):
    graph, chunks, keys, entity_vectors, chunk_vectors = load_index()
    assert keys == list(graph.nodes)
    assert entity_vectors.shape[0] == len(keys)
    assert chunk_vectors.shape[0] == len(chunks)


def test_meta_records_how_the_index_was_built(built):
    meta = json.loads((config.INDEX_DIR / "meta.json").read_text())
    assert meta["chunk_method"] == "paragraph"
    assert meta["chunk_count"] == len(built.index[1])
    assert meta["chunk_overlap"] is None  # only the fixed method overlaps
    assert meta["chat_model"] == "fake-chat"


def test_meta_defaults_for_old_indexes(project):
    assert load_meta() == {"chunk_method": "paragraph"}


def test_loading_a_missing_index_exits(project):
    with pytest.raises(SystemExit):
        load_index()
