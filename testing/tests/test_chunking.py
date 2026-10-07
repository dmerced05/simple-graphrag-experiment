# step 1: chunking.py
import re

import pytest

import config
from chunking import METHODS, load_chunks, pack, sections_of, split_fixed, split_structure


def words(text):
    return re.findall(r"\S+", text)


@pytest.mark.parametrize("method", METHODS)
def test_every_method_returns_well_formed_chunks(project, method):
    chunks = load_chunks(method)
    assert chunks
    assert len({c["id"] for c in chunks}) == len(chunks), "chunk ids must be unique"
    for c in chunks:
        assert set(c) == {"id", "source", "text", "method"}
        assert c["method"] == method
        assert c["text"].strip()
        assert c["id"].split("#")[0] == c["source"].rsplit(".", 1)[0]


@pytest.mark.parametrize("method", METHODS)
def test_ids_count_up_from_zero_per_file(project, method):
    by_file = {}
    for c in load_chunks(method):
        stem, n = c["id"].split("#")
        by_file.setdefault(stem, []).append(int(n))
    for numbers in by_file.values():
        assert numbers == list(range(len(numbers)))


def test_default_method_comes_from_config(project, monkeypatch):
    monkeypatch.setattr(config, "CHUNK_METHOD", "structure")
    assert {c["method"] for c in load_chunks()} == {"structure"}


def test_unknown_method_exits(project):
    with pytest.raises(SystemExit):
        load_chunks("nope")


def test_empty_data_folder_exits(project):
    for file in project.data.iterdir():
        file.unlink()
    with pytest.raises(SystemExit):
        load_chunks()


# ---------- paragraph (original) ----------

def test_paragraph_method_is_unchanged(project):
    # pinned on purpose: if this changes, every existing extraction cache misses and gets rebuilt
    chunks = load_chunks("paragraph")
    assert [c["id"] for c in chunks] == ["incidents#0", "incidents#1", "incidents#2", "incidents#3",
                                         "team#0", "team#1", "team#2", "vendors#0", "vendors#1"]
    assert chunks[0]["text"] == "# Incident log, spring quarter"  # the known heading-only chunk


def test_paragraph_boundary_is_exact(monkeypatch):
    # two 100-char paragraphs fit a 200 limit (the "\n\n" between them isn't counted), but not a 199 limit
    from chunking import split_paragraphs
    text = "a" * 100 + "\n\n" + "b" * 100
    monkeypatch.setattr(config, "CHUNK_CHARS", 200)
    assert len(split_paragraphs(text)) == 1
    monkeypatch.setattr(config, "CHUNK_CHARS", 199)
    assert len(split_paragraphs(text)) == 2


def test_paragraph_packs_small_files_into_one_chunk(project, monkeypatch):
    monkeypatch.setattr(config, "CHUNK_CHARS", 10_000)
    assert len(load_chunks("paragraph")) == 3


# ---------- level 1: fixed ----------

def test_fixed_respects_size(project):
    assert all(len(c["text"]) <= config.CHUNK_CHARS for c in load_chunks("fixed"))


def test_fixed_neighbors_overlap(project):
    chunks = load_chunks("fixed")
    for a, b in zip(chunks, chunks[1:]):
        if a["source"] == b["source"]:
            assert b["text"][:15] in a["text"], f"{b['id']} should start inside {a['id']}"


def test_fixed_never_chops_words_and_covers_everything(project):
    text = re.sub(r"\s+", " ", (project.data / "vendors.md").read_text()).strip()
    pieces = split_fixed(text)
    doc_words = set(words(text))
    seen = set()
    for piece in pieces:
        assert set(words(piece)) <= doc_words, "a word got cut in half"
        seen |= set(words(piece))
    assert seen == doc_words, "some words didn't land in any chunk"


def test_fixed_finishes_even_when_overlap_is_too_big(project, monkeypatch):
    monkeypatch.setattr(config, "CHUNK_OVERLAP", config.CHUNK_CHARS * 2)
    assert split_fixed("word " * 500)  # would loop forever without the overlap cap


# ---------- level 2: structure ----------

def test_structure_chunks_start_with_their_heading(project):
    for c in load_chunks("structure"):
        heading = (project.data / c["source"]).read_text().splitlines()[0].lstrip("# ")
        assert c["text"].startswith(heading + "\n\n")


def test_structure_drops_heading_only_chunks(project):
    for c in load_chunks("structure"):
        assert "\n\n" in c["text"], f"{c['id']} is just a heading"
        assert not c["text"].startswith("#")


def test_structure_tracks_nested_headings():
    text = "# Top\n\n## Sub\n\nfirst fact.\n\n## Other\n\nsecond fact.\n\n# Next\n\nthird fact."
    assert [path for path, _ in sections_of(text)] == [["Top", "Sub"], ["Top", "Other"], ["Next"]]
    assert split_structure(text)[0] == "Top > Sub\n\nfirst fact."


def test_structure_splits_long_paragraphs_into_sentences(project, monkeypatch):
    monkeypatch.setattr(config, "CHUNK_CHARS", 60)
    paragraph = " ".join(f"Sentence number {n} is here." for n in range(10))
    pieces = split_structure(f"# H\n\n{paragraph}")
    assert len(pieces) > 1
    for piece in pieces:
        assert len(piece.split("\n\n", 1)[1]) <= 60


def test_structure_works_without_headings():
    assert split_structure("just a plain paragraph.") == ["just a plain paragraph."]


def test_pack_glues_until_the_limit():
    assert pack(["aaaa", "bbbb", "cccc"], limit=10) == ["aaaa\n\nbbbb", "cccc"]
    assert pack(["x" * 50], limit=10) == ["x" * 50]  # oversized pieces stay whole
