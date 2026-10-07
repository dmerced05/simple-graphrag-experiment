# step 3: knowledge_graph.py
import pytest

from knowledge_graph import build_graph, graph_from_json, graph_to_json, normalize_name


@pytest.mark.parametrize("name, key", [
    ("The Payments API", "payments api"),
    ("payments  API!", "payments api"),
    ("Juniper Labs", "juniper labs"),
    ("third-party", "third party"),
    ("Theodore", "theodore"),  # only a leading "the " word is dropped
])
def test_normalize_name(name, key):
    assert normalize_name(name) == key


def make_graph():
    chunks = [{"id": "a#0"}, {"id": "b#0"}]
    extractions = {
        "a#0": {"entities": [{"name": "The Payments API", "type": "system", "description": "handles invoices"},
                             {"name": "Atlas", "type": "team"}],
                "relations": [{"source": "Atlas", "relation": "owns", "target": "payments api"},
                              {"source": "Atlas", "relation": "is", "target": "Atlas"}]},
        "b#0": {"entities": [{"name": "payments API", "type": "other", "description": "handles invoices"}],
                "relations": [{"source": "Redwood", "relation": "used by", "target": "Payments API"}]},
    }
    return build_graph(chunks, extractions)


def test_name_variants_merge_into_one_node():
    graph = make_graph()
    node = graph.nodes["payments api"]
    assert node["name"] == "The Payments API"   # first spelling wins
    assert node["type"] == "system"             # first type wins
    assert node["descriptions"] == ["handles invoices"]  # duplicates dropped
    assert node["chunks"] == ["a#0", "b#0"]


def test_relations_create_missing_endpoints_as_other():
    assert make_graph().nodes["redwood"]["type"] == "other"


def test_self_loops_are_dropped():
    assert not make_graph().has_edge("atlas", "atlas")


def test_edges_remember_their_chunk():
    graph = make_graph()
    assert [d["chunk"] for _, _, d in graph.edges("atlas", data=True)] == ["a#0"]
    assert "b#0" in graph.nodes["redwood"]["chunks"]


def test_json_round_trip():
    graph = make_graph()
    again = graph_from_json(graph_to_json(graph))
    assert dict(again.nodes(data=True)) == dict(graph.nodes(data=True))
    assert sorted(again.edges(data="relation")) == sorted(graph.edges(data="relation"))
