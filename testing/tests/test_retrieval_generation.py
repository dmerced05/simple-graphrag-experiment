# steps 7 and 8: retrieval.py and generation.py
import pytest

from generation import ANSWER_PROMPT, run_query
from retrieval import build_context

QUESTION = "Which vendor maintains the library that caused the outage in the system Priya Raman's team owns?"


def context_for(built, mode, **options):
    settings = {"hops": 2, "top_entities": 5, "top_chunks": 3, "budget": 300, **options}
    return build_context(QUESTION, built.index, mode, **settings)


@pytest.mark.parametrize("mode", ["vector", "graph", "wiki"])
def test_every_mode_reports_timings(built, mode):
    _, trace = context_for(built, mode)
    assert trace["embed_ms"] >= 0 and trace["retrieve_ms"] >= 0


def test_vector_mode_is_just_chunks(built):
    context, trace = context_for(built, "vector", top_chunks=2)
    assert len(trace["chunks"]) == 2 and trace["facts"] == []
    assert context.startswith("Source passages:")


def test_graph_mode_adds_cited_facts(built):
    context, trace = context_for(built, "graph")
    chunk_ids = {c["id"] for c in built.index[1]}
    assert trace["facts"] and context.startswith("Facts from the knowledge graph:")
    for fact in trace["facts"]:
        assert fact.rsplit("[", 1)[1].rstrip("]") in chunk_ids
    assert "Priya Raman" in trace["seeds"]  # named in the question


def test_wiki_mode_sends_no_raw_chunks(built):
    context, trace = context_for(built, "wiki")
    assert trace["chunks"] == [] and trace["pages"]
    assert context.startswith("Wiki pages:")
    assert "Source passages" not in context


def test_run_query_sends_context_and_question(built):
    result = run_query(QUESTION, built.index, "graph")
    prompt = built.fake.chat_calls[-1]["prompt"]
    assert prompt.startswith(ANSWER_PROMPT.split("<<CONTEXT>>")[0])
    assert prompt.rstrip().endswith(QUESTION)
    assert result["answer"] == "Juniper Labs [vendors#0]"


def test_run_query_metrics(built):
    built.fake.chat_load_ns, built.fake.embed_load_ns = 4_000_000, 1_000_000
    m = run_query(QUESTION, built.index, "wiki")["metrics"]
    for key in ("embed_ms", "retrieve_ms", "llm_ms", "total_ms", "warm_ms", "prompt_tokens", "output_tokens", "context_chars"):
        assert key in m
    assert m["load_ms"] == 4.0 and m["embed_load_ms"] == 1.0
    assert m["warm_ms"] == pytest.approx(m["total_ms"] - 5.0)  # loads are subtracted
