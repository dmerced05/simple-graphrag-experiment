# OKF (Open Knowledge Format) wiki: compile the knowledge graph into markdown pages, then retrieve from them
# spec: https://github.com/GoogleCloudPlatform/knowledge-catalog/tree/main/okf  (v0.1, only `type` is required)
#
# level 1: export_okf() writes one page per entity, an index.md per folder, and appends to log.md
# level 2: select_pages() answers from the compiled pages under a token budget, no raw chunks

import re
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")  # markdown link -> keep the text only


def slugify(text):
    slug = re.sub(r"[^\w\s-]", "", str(text).lower())
    return re.sub(r"[\s_]+", "-", slug).strip("-") or "untitled"


def estimate_tokens(text):
    # rough rule of thumb (~4 chars per token); good enough for budgeting and comparing modes
    return max(1, len(text) // 4)


def _frontmatter(fields):
    return "---\n" + yaml.safe_dump(fields, sort_keys=False, allow_unicode=True, width=1000) + "---\n"


# ---------- level 1: export ----------

def export_okf(graph, chunks, wiki_dir, chat_model=""):
    wiki_dir = Path(wiki_dir)
    wiki_dir.mkdir(parents=True, exist_ok=True)
    _clear_old_pages(wiki_dir)
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    source_file = {c["id"]: c["source"] for c in chunks}

    # give every entity a stable path like /person/priya-raman.md
    paths, taken = {}, set()
    for key, data in graph.nodes(data=True):
        folder = slugify(data["type"])
        slug, n = slugify(data["name"]), 2
        while f"{folder}/{slug}" in taken:
            slug, n = f"{slugify(data['name'])}-{n}", n + 1
        taken.add(f"{folder}/{slug}")
        paths[key] = f"/{folder}/{slug}.md"

    def link(key):
        return f"[{graph.nodes[key]['name']}]({paths[key]})"

    def bold(key):
        return f"**{graph.nodes[key]['name']}**"

    for key, data in graph.nodes(data=True):
        # I write relations as full triples on both ends so retrieval can drop the duplicate copy later
        outgoing = [f"- {bold(key)} {d['relation']} → {link(v)} [{d['chunk']}]"
                    for _, v, d in graph.out_edges(key, data=True)]
        incoming = [f"- {link(u)} {d['relation']} → {bold(key)} [{d['chunk']}]"
                    for u, _, d in graph.in_edges(key, data=True)]
        description = "; ".join(data["descriptions"])  # some entities get no description from extraction

        body = [f"# {data['name']}", ""] + ([description, ""] if description else [])
        if outgoing:
            body += ["## Relations", *outgoing, ""]
        if incoming:
            body += ["## Referenced by", *incoming, ""]
        body += ["## Sources", *[f"- `{cid}` from data/{source_file.get(cid, '?')}" for cid in data["chunks"]], ""]
        text = "\n".join(body)

        fields = {
            "type": data["type"],  # the only field OKF requires
            "title": data["name"],
            **({"description": description} if description else {}),
            "resource": f"data/{source_file.get(data['chunks'][0], '')}" if data["chunks"] else "",
            "tags": [data["type"]],
            "timestamp": now,
            # custom fields (OKF allows any extra keys); these are seeds for the design doc's metadata planes
            "id": key,                                   # links the page back to its graph node + embedding
            "sources": data["chunks"],                   # provenance plane
            "degree": graph.degree(key),                 # graph structural plane
            "token_footprint": estimate_tokens(text),    # LLM/context plane
        }
        page = wiki_dir / paths[key].lstrip("/")
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text(_frontmatter(fields) + "\n" + text, encoding="utf-8")

    _write_indexes(graph, paths, wiki_dir, now)
    duplicates = find_possible_duplicates(graph)
    _append_log(wiki_dir, now, graph, chat_model, duplicates)
    return len(paths), duplicates


def _clear_old_pages(wiki_dir):
    # rebuild from scratch each time, but keep log.md since it's the history
    for page in wiki_dir.rglob("*.md"):
        if page.resolve() != (wiki_dir / "log.md").resolve():
            page.unlink()
    for folder in sorted((p for p in wiki_dir.rglob("*") if p.is_dir()), reverse=True):
        if not any(folder.iterdir()):
            folder.rmdir()


def _write_indexes(graph, paths, wiki_dir, now):
    by_folder = {}
    for key, path in paths.items():
        by_folder.setdefault(path.split("/")[1], []).append(key)

    root = ["# Knowledge wiki", "", f"{graph.number_of_nodes()} pages compiled from the source documents.", ""]
    for folder, keys in sorted(by_folder.items()):
        root.append(f"- [{folder}](/{folder}/index.md) ({len(keys)})")
        lines = [f"# {folder}", ""]
        for key in sorted(keys, key=lambda k: graph.nodes[k]["name"].lower()):
            data = graph.nodes[key]
            summary = data["descriptions"][0] if data["descriptions"] else ""
            lines.append(f"- [{data['name']}]({paths[key]}) {('— ' + summary) if summary else ''}".rstrip())
        fields = {"type": "Index", "title": folder, "timestamp": now}
        (wiki_dir / folder / "index.md").write_text(_frontmatter(fields) + "\n" + "\n".join(lines) + "\n", encoding="utf-8")
    fields = {"type": "Index", "title": "Knowledge wiki", "timestamp": now}
    (wiki_dir / "index.md").write_text(_frontmatter(fields) + "\n" + "\n".join(root) + "\n", encoding="utf-8")


def find_possible_duplicates(graph):
    # cheap lint: "redwood" vs "redwood caching library" are probably the same thing but became two nodes.
    # I only flag them; merging automatically would be wrong too often (e.g. "Beacon" the team vs a product)
    names = {k: graph.nodes[k]["name"] for k in graph.nodes}
    pairs = []
    for a in names:
        for b in names:
            if a != b and len(a) > 3 and b.startswith(a + " "):
                pairs.append((names[a], names[b]))
    return pairs


def _append_log(wiki_dir, now, graph, chat_model, duplicates):
    log = wiki_dir / "log.md"
    if not log.exists():
        log.write_text(_frontmatter({"type": "Log", "title": "Compile log"}) + "\n# Compile log\n", encoding="utf-8")
    entry = [f"\n## {now}", f"- compiled {graph.number_of_nodes()} pages, {graph.number_of_edges()} relations"
             + (f" with {chat_model}" if chat_model else "")]
    if duplicates:
        entry.append("- possible duplicate entities (not merged):")
        entry += [f"  - {a} / {b}" for a, b in duplicates]
    with log.open("a", encoding="utf-8") as f:
        f.write("\n".join(entry) + "\n")


# ---------- level 2: retrieval from the wiki ----------

_cache = {}


def load_wiki(wiki_dir):
    # parse every page once and keep it in memory; reloads if the wiki was re-exported
    wiki_dir = Path(wiki_dir)
    log = wiki_dir / "log.md"
    stamp = log.stat().st_mtime if log.exists() else None
    if stamp is None:
        raise FileNotFoundError(f"No wiki in {wiki_dir}/. Run: python graphrag.py export-okf")
    if _cache.get("stamp") == stamp and _cache.get("dir") == wiki_dir:
        return _cache["pages"]

    pages = {}
    for path in wiki_dir.rglob("*.md"):
        if path.name in ("index.md", "log.md"):
            continue
        raw = path.read_text(encoding="utf-8")
        _, front, body = raw.split("---\n", 2)
        fields = yaml.safe_load(front)
        rel_path = "/" + path.relative_to(wiki_dir).as_posix()
        pages[rel_path] = {
            "fields": fields,
            "links": set(re.findall(r"\]\((/[^)]+\.md)\)", body)),
            "compact": _compact(fields, body),
        }
    _cache.update(stamp=stamp, dir=wiki_dir, pages=pages)
    return pages


def _compact(fields, body):
    # what actually goes into the prompt: title, description, relation triples. no link urls, no sources list
    lines = [f"### {fields['title']} ({fields['type']})", fields.get("description", "")]  # "" if none
    in_sources = False
    for line in body.splitlines():
        if line.startswith("## "):
            in_sources = line.startswith("## Sources")
            continue
        if line.startswith("- ") and not in_sources:
            lines.append(LINK.sub(r"\1", line).replace("**", ""))
    return lines


def select_pages(question, question_vector, pages, entity_keys, entity_vectors,
                 budget=300, hops=2, top_entities=5, hop_penalty=0.15):
    # retrieval as a tiny knapsack: score pages, then fill a fixed token budget greedily,
    # skipping relation lines that an earlier page already contributed (the redundancy term in the design doc)
    by_id = {p["fields"]["id"]: path for path, p in pages.items()}
    similarity = {}
    if len(entity_keys):
        scores = entity_vectors @ question_vector
        similarity = {by_id[k]: float(s) for k, s in zip(entity_keys, scores) if k in by_id}

    seeds = sorted(similarity, key=similarity.get, reverse=True)[:top_entities]
    lowered = question.lower()
    named = set()  # pages the question names outright always go first
    for path, page in pages.items():
        title = str(page["fields"]["title"]).lower()
        if len(title) > 3 and title in lowered:
            named.add(path)
            if path not in seeds:
                seeds.append(path)

    # walk the wiki's own links (both directions are on every page) to find nearby pages
    distance = {s: 0 for s in seeds}
    queue = deque(seeds)
    while queue:
        path = queue.popleft()
        if distance[path] >= hops:
            continue
        for neighbor in pages[path]["links"]:
            if neighbor in pages and neighbor not in distance:
                distance[neighbor] = distance[path] + 1
                queue.append(neighbor)

    ranked = sorted(distance, key=lambda p: (p in named) + similarity.get(p, 0.0) - hop_penalty * distance[p],
                    reverse=True)

    picked, seen_lines, used = [], set(), 0
    for path in ranked:
        page = pages[path]
        header, description, *relations = page["compact"]
        fresh = [line for line in relations if line not in seen_lines]
        block = "\n".join(line for line in [header, description, *fresh] if line)
        cost = estimate_tokens(block)
        if picked and used + cost > budget:
            continue  # doesn't fit; a smaller, lower-ranked page might
        picked.append((path, block, fresh))
        seen_lines.update(fresh)
        used += cost

    trace = {
        "seeds": [pages[s]["fields"]["title"] for s in seeds],
        "pages_considered": len(ranked),
        "pages": [pages[p]["fields"]["title"] for p, _, _ in picked],
        "tokens_used": used,
        "budget": budget,
        "facts": [line for _, _, fresh in picked for line in fresh],
        "sources": sorted({cid for p, _, _ in picked for cid in pages[p]["fields"].get("sources", [])}),
    }
    context = "Wiki pages:\n\n" + "\n\n".join(block for _, block, _ in picked)
    return context, trace
