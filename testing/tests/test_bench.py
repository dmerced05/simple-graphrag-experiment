# the benchmark itself (testing/bench.py)
import json
from collections import Counter
from types import SimpleNamespace

import pytest

from testing import bench

ROOT_DATA = bench.BENCH_DIR.parent.parent / "data"


@pytest.fixture
def results_dir(tmp_path, monkeypatch):
    folder = tmp_path / "results"
    monkeypatch.setattr(bench, "RESULTS_DIR", folder)
    return folder


def test_is_correct():
    assert bench.is_correct("It's juniper labs.", ["Juniper Labs"])
    assert bench.is_correct("Halcyon makes it", ["Halcyon Software", "Halcyon"])
    assert not bench.is_correct("I don't know", ["Juniper Labs"])


def test_evidence_recall_counts_chunks_facts_and_wiki_sources():
    trace = {"chunks": ["team#0"], "facts": ["Atlas --owns--> payments API [incidents#1]"], "sources": ["vendors#0"]}
    assert bench.evidence_recall(trace, ["team", "incidents", "vendors"]) == 1.0
    assert bench.evidence_recall({"chunks": ["team#0"]}, ["team", "vendors"]) == 0.5


def test_median_and_p95():
    assert bench.median([3, 1, 2]) == 2 and bench.median([]) == 0.0
    assert bench.p95(list(range(1, 101))) == 95
    assert bench.p95([5]) == 5


def test_order_effect_compares_within_questions():
    rows = [{"id": "a", "order": 1, "warm_ms": 10}, {"id": "a", "order": 2, "warm_ms": 10},
            {"id": "b", "order": 1, "warm_ms": 100}, {"id": "b", "order": 2, "warm_ms": 90}]
    assert bench._order_effect(rows) == 5  # median of (0, 10)


def test_unknown_option_is_rejected():
    with pytest.raises(TypeError):
        bench.default_args(nonsense=1)


def test_real_question_file_is_valid():
    questions = json.loads(bench.QUESTIONS_FILE.read_text())
    assert len({q["id"] for q in questions}) == len(questions)
    for q in questions:
        assert q["type"] in ("simple", "multi-hop") and q["answers"] and q["question"].endswith("?")
        for source in q["sources"]:
            assert (ROOT_DATA / f"{source}.md").exists(), f"{q['id']} cites missing data/{source}.md"


def test_mode_order_is_balanced(built, results_dir):
    rows, _ = bench.run_bench(runs=3, retrieval_only=True)
    slots = Counter((r["mode"], r["order"]) for r in rows)
    modes = {r["mode"] for r in rows}
    assert set(slots) == {(m, n) for m in modes for n in range(1, len(modes) + 1)}, "every mode in every slot"
    assert len(set(slots.values())) == 1, f"each mode should take each slot equally often: {slots}"


def test_full_run_scores_and_saves(built, results_dir, capsys):
    rows, summary = bench.run_bench(runs=1, only="outage-vendor,beacon-lead")
    assert set(summary) == {"graph", "vector", "wiki"}
    by_id = {(r["id"], r["mode"]): r["correct"] for r in rows}
    assert by_id[("outage-vendor", "graph")] is True
    assert by_id[("beacon-lead", "graph")] is False  # the fake model only ever knows Juniper Labs
    saved = json.loads(next(results_dir.glob("*.json")).read_text())
    assert saved["config"]["chunk_method"] == "paragraph"
    assert len(saved["rows"]) == len(rows)
    assert "KPI" in capsys.readouterr().out


def test_runs_in_the_same_second_dont_overwrite(built, results_dir):
    args = bench.default_args(runs=1)
    bench.save([{"a": 1}], {}, args)
    bench.save([{"a": 2}], {}, args)
    assert len(list(results_dir.glob("*.json"))) == 2
