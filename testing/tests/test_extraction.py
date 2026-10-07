# step 2: extraction.py
import json

import config
from chunking import load_chunks
from extraction import extract, extract_all


def test_extract_keeps_valid_items_and_drops_broken_ones(fake_ollama):
    fake_ollama.queued = [json.dumps({
        "entities": [{"name": "Redwood", "type": "product"}, {"type": "no name"}, "not a dict"],
        "relations": [{"source": "Redwood", "relation": "made by", "target": "Juniper Labs"},
                      {"source": "Redwood", "relation": "missing target"}],
    })]
    result = extract("Redwood is made by Juniper Labs.")
    assert [e["name"] for e in result["entities"]] == ["Redwood"]
    assert len(result["relations"]) == 1
    assert fake_ollama.chat_calls[0]["format"] == "json"


def test_extract_retries_once_then_gives_up(fake_ollama):
    fake_ollama.queued = ["not json", "still not json"]
    assert extract("anything") == {"entities": [], "relations": []}
    assert len(fake_ollama.chat_calls) == 2


def test_extract_recovers_on_the_retry(fake_ollama):
    fake_ollama.queued = ["oops", json.dumps({"entities": [{"name": "Atlas"}], "relations": []})]
    assert extract("Atlas")["entities"] == [{"name": "Atlas"}]


def test_extract_all_caches_by_chunk(project):
    chunks = load_chunks()
    config.INDEX_DIR.mkdir()
    first = extract_all(chunks)
    calls = len(project.fake.chat_calls)
    assert calls == len(chunks)
    assert extract_all(chunks) == first
    assert len(project.fake.chat_calls) == calls, "second run should be all cache hits"


def test_changing_the_model_re_extracts(project, monkeypatch):
    chunks = load_chunks()
    config.INDEX_DIR.mkdir()
    extract_all(chunks)
    monkeypatch.setattr(config, "CHAT_MODEL", "another-model")
    extract_all(chunks)
    assert len(project.fake.chat_calls) == 2 * len(chunks)
