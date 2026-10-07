# step 6: okf.py (wiki export + wiki-mode retrieval)
import re

import networkx as nx
import pytest
import yaml

import config
import okf
from embedding import embed


def pages(wiki_dir):
    return [p for p in wiki_dir.rglob("*.md") if p.name not in ("index.md", "log.md")]


def frontmatter(path):
    return yaml.safe_load(path.read_text().split("---\n", 2)[1])


def test_one_page_per_entity(built):
    graph = built.index[0]
    found = {frontmatter(p)["id"] for p in pages(config.WIKI_DIR)}
    assert found == set(graph.nodes)


def test_pages_follow_okf(built):
    for page in pages(config.WIKI_DIR):
        fields = frontmatter(page)
        assert fields["type"], "type is the one field OKF requires"
        assert fields["title"] and fields["sources"]
        assert fields["token_footprint"] > 0
        assert page.parent.name == okf.slugify(fields["type"])  # grouped by type


def test_every_link_points_at_a_real_page(built):
    for page in config.WIKI_DIR.rglob("*.md"):
        for target in re.findall(r"\]\((/[^)]+\.md)\)", page.read_text()):
            assert (config.WIKI_DIR / target.lstrip("/")).exists(), f"{page.name} links to missing {target}"


def test_indexes_exist_for_root_and_each_folder(built):
    assert (config.WIKI_DIR / "index.md").exists()
    for folder in {p.parent for p in pages(config.WIKI_DIR)}:
        assert (folder / "index.md").exists()


def test_rebuilding_keeps_the_log_and_drops_stale_pages(built):
    stale = config.WIKI_DIR / "other" / "stale.md"
    stale.write_text("---\ntype: other\n---\nold")
    okf.export_okf(built.index[0], built.index[1], config.WIKI_DIR)
    assert not stale.exists()
    assert (config.WIKI_DIR / "log.md").read_text().count("\n## ") == 2  # one entry per compile


def test_entities_without_a_description_get_none(tmp_path):
    # no made-up filler text: the field is left out and the prompt line is empty
    graph = nx.MultiDiGraph()
    graph.add_node("x", name="X", type="other", descriptions=[], chunks=["a#0"])
    okf.export_okf(graph, [{"id": "a#0", "source": "a.md"}], tmp_path)
    page = tmp_path / "other" / "x.md"
    assert "description" not in frontmatter(page)
    assert okf.load_wiki(tmp_path)["/other/x.md"]["compact"][1] == ""


def test_slug_collisions_get_a_suffix(tmp_path):
    graph = nx.MultiDiGraph()
    for key in ("x y", "x_y"):  # different graph keys, same slug
        graph.add_node(key, name="X Y", type="other", descriptions=["d"], chunks=["a#0"])
    okf.export_okf(graph, [{"id": "a#0", "source": "a.md"}], tmp_path)
    assert sorted(p.name for p in (tmp_path / "other").glob("x-y*.md")) == ["x-y-2.md", "x-y.md"]


def test_possible_duplicates_are_flagged():
    graph = nx.MultiDiGraph()
    for key, name in [("redwood", "Redwood"), ("redwood caching library", "Redwood caching library"), ("atlas", "Atlas")]:
        graph.add_node(key, name=name)
    assert okf.find_possible_duplicates(graph) == [("Redwood", "Redwood caching library")]


@pytest.mark.parametrize("text, slug", [("Priya Raman", "priya-raman"), ("payments API!", "payments-api"), ("???", "untitled")])
def test_slugify(text, slug):
    assert okf.slugify(text) == slug


def test_missing_wiki_raises(project):
    with pytest.raises(FileNotFoundError):
        okf.load_wiki(config.WIKI_DIR)


# ---------- wiki-mode retrieval ----------

def select(built, question, budget=300):
    _, _, keys, entity_vectors, _ = built.index
    wiki = okf.load_wiki(config.WIKI_DIR)
    return okf.select_pages(question, embed([question])[0], wiki, keys, entity_vectors, budget=budget)


def test_selection_stays_within_budget(built):
    for budget in (40, 100, 300):
        _, trace = select(built, "Which vendor maintains Redwood?", budget)
        assert trace["tokens_used"] <= budget or len(trace["pages"]) == 1


def test_tiny_budget_still_returns_one_page(built):
    _, trace = select(built, "Who leads Atlas?", budget=1)
    assert len(trace["pages"]) == 1


def test_named_pages_come_first(built):
    # the question is mostly about caching and vendors, so similarity alone would pick another page first;
    # naming Atlas outright has to win anyway
    question = "small vendor caching library memory leak session data based in a city, and Atlas"
    _, trace = select(built, question, budget=1)
    assert trace["pages"] == ["Atlas"]


def test_relation_lines_are_not_repeated(built):
    context, trace = select(built, "Tell me about Redwood and Juniper Labs")
    assert len(trace["facts"]) == len(set(trace["facts"]))
    assert "](" not in context, "link urls should be stripped from the prompt"
