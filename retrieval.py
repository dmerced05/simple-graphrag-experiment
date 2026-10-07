# step 7 (query time): embed the question and build the context for one of the three modes
#   vector: the most similar chunks
#   graph:  facts from a walk around the knowledge graph + the chunks they came from
#   wiki:   compiled OKF pages packed into a token budget (see okf.py)
import time

import networkx as nx
import numpy as np

import config
import okf
from embedding import embed


def build_context(question, index, mode, hops, top_entities, top_chunks, budget=300):
    graph, chunks, entity_keys, entity_vectors, chunk_vectors = index
    start = time.perf_counter()
    embed_stats = {}
    question_vector = embed([question], stats=embed_stats)[0]
    trace = {"embed_ms": (time.perf_counter() - start) * 1000, "embed_load_ms": embed_stats.get("load_ms", 0.0)}
    start = time.perf_counter()  # everything after this is pure retrieval, no model calls
    chunk_scores = chunk_vectors @ question_vector

    if mode == "wiki":
        # compiled-wiki mode: only the OKF pages go to the model, packed into a fixed token budget
        pages = okf.load_wiki(config.WIKI_DIR)
        context, wiki_trace = okf.select_pages(question, question_vector, pages, entity_keys, entity_vectors,
                                               budget=budget, hops=hops, top_entities=top_entities)
        trace.update(wiki_trace)
        trace["chunks"] = []
        trace["retrieve_ms"] = (time.perf_counter() - start) * 1000
        return context, trace

    if mode == "vector":
        # plain RAG baseline: just the most similar chunks, no graph at all
        best = np.argsort(chunk_scores)[::-1][:top_chunks]
        picked_chunks = [chunks[i] for i in best]
        facts = []
    else:
        # 1) entry points: entities most similar to the question, plus any named outright
        seeds = []
        if len(entity_keys):
            for i in np.argsort(entity_vectors @ question_vector)[::-1][:top_entities]:
                seeds.append(entity_keys[i])
        lowered = question.lower()
        for key, data in graph.nodes(data=True):
            if key not in seeds and len(key) > 3 and data["name"].lower() in lowered:
                seeds.append(key)

        # 2) walk outward from the seeds, ignoring edge direction
        neighborhood = set()
        for seed in seeds:
            neighborhood |= set(nx.ego_graph(graph, seed, radius=hops, undirected=True).nodes)

        # 3) facts = every edge inside that neighborhood, written as readable triples
        facts = []
        for u, v, data in graph.subgraph(neighborhood).edges(data=True):
            facts.append(f"{graph.nodes[u]['name']} --{data['relation']}--> {graph.nodes[v]['name']} [{data['chunk']}]")
        facts = sorted(set(facts))[:60]

        # 4) source passages: chunks attached to those entities, best matches first
        candidate_ids = {cid for key in neighborhood for cid in graph.nodes[key]["chunks"]}
        candidates = [c for c in chunks if c["id"] in candidate_ids]
        position = {c["id"]: i for i, c in enumerate(chunks)}
        candidates.sort(key=lambda c: chunk_scores[position[c["id"]]], reverse=True)
        picked_chunks = candidates[:top_chunks]
        trace["seeds"] = [graph.nodes[s]["name"] for s in seeds]
        trace["neighborhood_size"] = len(neighborhood)

    parts = []
    if facts:
        parts.append("Facts from the knowledge graph:\n" + "\n".join(facts))
    parts.append("Source passages:\n" + "\n\n".join(f"[{c['id']}]\n{c['text']}" for c in picked_chunks))
    trace["facts"] = facts
    trace["chunks"] = [c["id"] for c in picked_chunks]
    trace["retrieve_ms"] = (time.perf_counter() - start) * 1000
    return "\n\n".join(parts), trace
