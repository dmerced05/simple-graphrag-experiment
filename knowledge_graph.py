# step 3: merge every chunk's extraction into one knowledge graph (and save/load it as JSON)
import re

import networkx as nx


def normalize_name(name):
    # cheap entity resolution: "The Payments API" and "payments api" become the same node
    key = re.sub(r"[^\w\s]", " ", str(name).lower())
    key = re.sub(r"\s+", " ", key).strip()
    return re.sub(r"^the ", "", key)


def build_graph(chunks, extractions):
    graph = nx.MultiDiGraph()

    def ensure_node(name, entity_type="other"):
        key = normalize_name(name)
        if key and key not in graph:
            graph.add_node(key, name=str(name).strip(), type=entity_type, descriptions=[], chunks=[])
        return key

    for chunk in chunks:
        result = extractions[chunk["id"]]
        for entity in result["entities"]:
            key = ensure_node(entity["name"], entity.get("type", "other"))
            if not key:
                continue
            node = graph.nodes[key]
            description = str(entity.get("description", "")).strip()
            if description and description not in node["descriptions"]:
                node["descriptions"].append(description)
            if chunk["id"] not in node["chunks"]:
                node["chunks"].append(chunk["id"])

        for relation in result["relations"]:
            source = ensure_node(relation["source"])
            target = ensure_node(relation["target"])
            if source and target and source != target:
                graph.add_edge(source, target, relation=str(relation.get("relation", "related to")), chunk=chunk["id"])
                # make sure both ends point back at the chunk that mentioned them
                for key in (source, target):
                    if chunk["id"] not in graph.nodes[key]["chunks"]:
                        graph.nodes[key]["chunks"].append(chunk["id"])
    return graph


def graph_to_json(graph):
    return {
        "nodes": [{"key": k, **data} for k, data in graph.nodes(data=True)],
        "edges": [{"source": u, "target": v, **data} for u, v, data in graph.edges(data=True)],
    }


def graph_from_json(blob):
    graph = nx.MultiDiGraph()
    for node in blob["nodes"]:
        key = node.pop("key")
        graph.add_node(key, **node)
    for edge in blob["edges"]:
        graph.add_edge(edge.pop("source"), edge.pop("target"), **edge)
    return graph
