# end to end: the graphrag.py commands, run in-process against the fake Ollama
import sys

import pytest

import config
import graphrag
from testing import bench


def run(monkeypatch, capsys, *argv):
    monkeypatch.setattr(sys, "argv", ["graphrag.py", *argv])
    graphrag.main()
    return capsys.readouterr().out


def test_index_then_ask_in_every_mode(project, monkeypatch, capsys):
    out = run(monkeypatch, capsys, "index")
    assert "Index saved" in out and "OKF wiki pages" in out
    for mode in ("graph", "vector", "wiki"):
        out = run(monkeypatch, capsys, "ask", "Who maintains Redwood? Juniper Labs?", "--mode", mode,
                  "--show-context", "--timing")
        assert "--- retrieval ---" in out and f"[{mode}] total" in out


@pytest.mark.parametrize("method", ["paragraph", "fixed", "structure"])
def test_index_with_each_chunk_method(project, monkeypatch, capsys, method):
    out = run(monkeypatch, capsys, "index", "--chunk-method", method)
    assert f"method={method}" in out
    assert f"method={method}" in run(monkeypatch, capsys, "stats")


def test_chunks_preview_needs_no_index_or_llm(project, monkeypatch, capsys):
    out = run(monkeypatch, capsys, "chunks", "--chunk-method", "fixed", "--chunk-chars", "120")
    assert "method=fixed, chunk_chars=120, overlap=50" in out
    assert project.fake.chat_calls == [] and not config.INDEX_DIR.exists()


def test_exports(built, monkeypatch, capsys):
    run(monkeypatch, capsys, "export")
    assert (config.INDEX_DIR / "graph.graphml").exists()
    assert "OKF wiki pages" in run(monkeypatch, capsys, "export-okf")


def test_bench_command(built, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(bench, "RESULTS_DIR", tmp_path / "results")
    out = run(monkeypatch, capsys, "bench", "--retrieval-only", "--runs", "1", "--modes", "graph,wiki")
    assert "evidence recall" in out


def test_asking_before_indexing_explains_what_to_do(project, monkeypatch, capsys):
    with pytest.raises(SystemExit) as stop:
        run(monkeypatch, capsys, "ask", "anything")
    assert "graphrag.py index" in str(stop.value)
