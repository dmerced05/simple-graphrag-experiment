# simple local GraphRAG: command line entry point. each pipeline step lives in its own file:
#
#   index time:  chunking.py -> extraction.py -> knowledge_graph.py -> embedding.py -> index_store.py -> okf.py (wiki)
#   query time:  retrieval.py (vector / graph / wiki context) -> generation.py (answer + metrics)
#   settings:    config.py        tests + benchmark: testing/ (bench.py, bench.ipynb, tests/)
#
# usage:
#   python graphrag.py index
#   python graphrag.py ask "your question"            (graph mode)
#   python graphrag.py ask "your question" --mode vector   (plain RAG baseline)
#   python graphrag.py ask "your question" --mode wiki     (answer from the compiled OKF wiki only)
#   python graphrag.py ask                             (interactive)
#   python graphrag.py stats | export | export-okf
#   python graphrag.py chunks --chunk-method structure   (preview chunking, no LLM calls)
#   python graphrag.py bench                          (compare modes, see testing/bench.py)

import argparse
import sys

import httpx
import networkx as nx
import numpy as np
import ollama

import config
import okf
from chunking import METHODS, load_chunks
from embedding import embed, entity_text
from extraction import extract_all
from generation import run_query
from index_store import load_index, load_meta, save_index
from knowledge_graph import build_graph
from retrieval import build_context  # noqa: F401  (re-exported so notebooks can call graphrag.build_context)

# re-exported so older code like graphrag.CHAT_MODEL keeps working; change settings in config.py or with env vars
from config import CHAT_MODEL, CHUNK_CHARS, DATA_DIR, EMBED_MODEL, INDEX_DIR, KEEP_ALIVE, WIKI_DIR  # noqa: F401


# ---------- index time: run the whole pipeline ----------

def cmd_index(args):
    apply_chunk_args(args)
    config.INDEX_DIR.mkdir(exist_ok=True)

    chunks = load_chunks()                                   # step 1: chunking.py
    print(f"Loaded {len(chunks)} chunks from {config.DATA_DIR}/ ({describe_chunking()})")

    print(f"Extracting entities and relations with {config.CHAT_MODEL}...")
    extractions = extract_all(chunks)                        # step 2: extraction.py

    graph = build_graph(chunks, extractions)                 # step 3: knowledge_graph.py
    print(f"Graph: {graph.number_of_nodes()} entities, {graph.number_of_edges()} relations")

    print(f"Embedding with {config.EMBED_MODEL}...")         # step 4: embedding.py
    entity_keys = list(graph.nodes)
    entity_vectors = embed([entity_text(graph.nodes[k]) for k in entity_keys]) if entity_keys else np.zeros((0, 1))
    chunk_vectors = embed([c["text"] for c in chunks])

    save_index(graph, chunks, entity_keys, entity_vectors, chunk_vectors)   # step 5: index_store.py
    print(f"Done. Index saved to {config.INDEX_DIR}/")
    write_wiki(graph, chunks)                                # step 6: okf.py


def apply_chunk_args(args):
    # command line flags override the env/config defaults for this run
    for flag, setting in (("chunk_method", "CHUNK_METHOD"), ("chunk_chars", "CHUNK_CHARS"), ("chunk_overlap", "CHUNK_OVERLAP")):
        value = getattr(args, flag, None)
        if value is not None:
            setattr(config, setting, value)


def describe_chunking():
    text = f"method={config.CHUNK_METHOD}, chunk_chars={config.CHUNK_CHARS}"
    return text + (f", overlap={config.CHUNK_OVERLAP}" if config.CHUNK_METHOD == "fixed" else "")


def cmd_chunks(args):
    # preview how a chunking method splits the documents, without touching the index or calling the LLM
    apply_chunk_args(args)
    chunks = load_chunks()
    sizes = [len(c["text"]) for c in chunks]
    for c in chunks:
        print(f"--- {c['id']}  ({len(c['text'])} chars)")
        print(c["text"] if args.full else c["text"][:300] + ("..." if len(c["text"]) > 300 else ""))
    print(f"\n{len(chunks)} chunks ({describe_chunking()}), "
          f"{min(sizes)}-{max(sizes)} chars, {sum(sizes)} chars total")


def write_wiki(graph, chunks):
    count, duplicates = okf.export_okf(graph, chunks, config.WIKI_DIR, config.CHAT_MODEL)
    print(f"Wrote {count} OKF wiki pages to {config.WIKI_DIR}/")
    if duplicates:
        print(f"  {len(duplicates)} possible duplicate entities (listed in {config.WIKI_DIR}/log.md), e.g. "
              + ", ".join(f"{a} / {b}" for a, b in duplicates[:3]))


# ---------- query time ----------

def answer_one(question, index, args):
    result = run_query(question, index, args.mode, args.hops, args.top_entities, args.top_chunks, args.budget)
    trace, m = result["trace"], result["metrics"]

    if args.show_context:
        print("\n--- retrieval ---")
        if "seeds" in trace:
            print(f"seed entities: {', '.join(trace['seeds'])}")
            if "neighborhood_size" in trace:
                print(f"neighborhood:  {trace['neighborhood_size']} entities, {len(trace['facts'])} facts")
            for fact in trace["facts"]:
                print(f"  {fact}")
        if "pages" in trace:
            print(f"wiki pages:    {', '.join(trace['pages'])}  ({trace['tokens_used']}/{trace['budget']} tokens, "
                  f"{trace['pages_considered']} considered)")
        else:
            print(f"passages:      {', '.join(trace['chunks'])}")
        print("-----------------\n")

    print(result["answer"])
    if args.timing:
        print(f"\n[{args.mode}] total {m['total_ms']:.0f} ms  (embed {m['embed_ms']:.0f}, retrieve {m['retrieve_ms']:.1f}, "
              f"llm {m['llm_ms']:.0f})  prompt {m['prompt_tokens']} tok, output {m['output_tokens']} tok")


def cmd_ask(args):
    index = load_index()
    if args.question:
        answer_one(args.question, index, args)
        return
    print(f"Interactive mode ({args.mode}). Empty line to quit.")
    while True:
        question = input("\n? ").strip()
        if not question:
            break
        answer_one(question, index, args)


# ---------- inspection and exports ----------

def cmd_stats(_args):
    graph, chunks = load_index()[:2]
    meta = load_meta()
    print(f"{len(chunks)} chunks (method={meta['chunk_method']}), "
          f"{graph.number_of_nodes()} entities, {graph.number_of_edges()} relations\n")
    print("Most connected entities:")
    for key, degree in sorted(graph.degree, key=lambda pair: pair[1], reverse=True)[:15]:
        data = graph.nodes[key]
        print(f"  {degree:3}  {data['name']}  ({data['type']})")


def cmd_export_okf(_args):
    graph, chunks = load_index()[:2]
    write_wiki(graph, chunks)


def cmd_export(_args):
    # GraphML opens in Gephi or yEd if I want to look at the graph visually
    graph = load_index()[0]
    flat = nx.MultiDiGraph()
    for key, data in graph.nodes(data=True):
        flat.add_node(key, name=data["name"], type=data["type"], description=" | ".join(data["descriptions"]))
    for u, v, data in graph.edges(data=True):
        flat.add_edge(u, v, relation=data["relation"], chunk=data["chunk"])
    out = config.INDEX_DIR / "graph.graphml"
    nx.write_graphml(flat, out)
    print(f"Wrote {out}")


def main():
    parser = argparse.ArgumentParser(description="Simple local GraphRAG")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_chunk_flags(command):
        command.add_argument("--chunk-method", choices=METHODS, help="default: CHUNK_METHOD env or 'paragraph'")
        command.add_argument("--chunk-chars", type=int, help="max chunk size in characters (default 200)")
        command.add_argument("--chunk-overlap", type=int, help="fixed method only: overlap in characters (default 50)")

    index = sub.add_parser("index", help="build the knowledge graph from DATA_DIR")
    add_chunk_flags(index)
    index.set_defaults(func=cmd_index)

    chunks = sub.add_parser("chunks", help="preview how the documents get chunked (no LLM calls)")
    add_chunk_flags(chunks)
    chunks.add_argument("--full", action="store_true", help="print whole chunks instead of the first 300 chars")
    chunks.set_defaults(func=cmd_chunks)

    ask = sub.add_parser("ask", help="ask a question (omit it for interactive mode)")
    ask.add_argument("question", nargs="?")
    ask.add_argument("--mode", choices=["graph", "vector", "wiki"], default="graph")
    ask.add_argument("--hops", type=int, default=2)
    ask.add_argument("--top-entities", type=int, default=5)
    ask.add_argument("--top-chunks", type=int, default=3)
    ask.add_argument("--budget", type=int, default=300, help="wiki mode: max context tokens")
    ask.add_argument("--show-context", action="store_true", help="print what was retrieved")
    ask.add_argument("--timing", action="store_true", help="print latency and token counts after the answer")
    ask.set_defaults(func=cmd_ask)

    from testing.bench import add_bench_parser  # lives in testing/ with the rest of the test tooling
    add_bench_parser(sub)

    sub.add_parser("stats", help="show graph size and most connected entities").set_defaults(func=cmd_stats)
    sub.add_parser("export", help="write index/graph.graphml").set_defaults(func=cmd_export)
    sub.add_parser("export-okf", help="write the OKF markdown wiki to WIKI_DIR").set_defaults(func=cmd_export_okf)

    args = parser.parse_args()
    try:
        args.func(args)
    except ollama.ResponseError as error:
        sys.exit(f"Ollama error: {error}\nDid you run: ollama pull {config.CHAT_MODEL} && ollama pull {config.EMBED_MODEL}?")
    except BrokenPipeError:
        pass  # output was piped into something like head; not an Ollama problem
    except FileNotFoundError as error:
        sys.exit(str(error))
    except (ConnectionError, httpx.ConnectError):
        sys.exit("Can't reach Ollama. Is the Ollama app running?")


if __name__ == "__main__":
    main()
