# benchmark: run the same questions through graph, vector and wiki mode and compare speed, cost and accuracy
# usage:
#   python graphrag.py bench                       (full run: retrieval + LLM answer, 3 runs per question)
#   python graphrag.py bench --runs 5
#   python graphrag.py bench --retrieval-only      (skip the LLM, just time retrieval; fast)
#   python graphrag.py bench --only outage-vendor  (one question)

import csv
import json
import math
import os
import statistics
import time
from datetime import datetime
from pathlib import Path

import ollama

import config
from embedding import embed
from generation import run_query
from index_store import load_index, load_meta
from retrieval import build_context

# anchored to this file so the benchmark finds its questions no matter where it's run from
BENCH_DIR = Path(__file__).resolve().parent / "benchmark"
QUESTIONS_FILE = BENCH_DIR / "questions.json"
RESULTS_DIR = BENCH_DIR / "results"


def add_bench_parser(sub):
    bench = sub.add_parser("bench", help="time and score graph vs vector mode on benchmark/questions.json")
    bench.add_argument("--runs", type=int, default=3, help="repeats per question per mode (more = steadier timings)")
    bench.add_argument("--modes", default="graph,vector,wiki")
    bench.add_argument("--questions", type=Path, default=QUESTIONS_FILE)
    bench.add_argument("--only", help="comma-separated question ids to run")
    bench.add_argument("--retrieval-only", action="store_true", help="skip the answer LLM call")
    bench.add_argument("--hops", type=int, default=2)
    bench.add_argument("--top-entities", type=int, default=5)
    bench.add_argument("--top-chunks", type=int, default=3)
    bench.add_argument("--budget", type=int, default=300, help="wiki mode: max context tokens")
    bench.set_defaults(func=cmd_bench)


def default_args(**options):
    # same defaults as the command line, so notebook and terminal runs are comparable
    import argparse
    parser = argparse.ArgumentParser()
    add_bench_parser(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["bench"])
    for key, value in options.items():
        if not hasattr(args, key):
            raise TypeError(f"unknown bench option: {key}")
        setattr(args, key, Path(value) if key == "questions" else value)
    return args


# ---------- scoring ----------

def is_correct(answer, expected):
    # crude but transparent: the answer counts if it mentions any accepted form of the expected answer
    lowered = answer.lower()
    return any(e.lower() in lowered for e in expected)


def evidence_recall(trace, gold_sources):
    # what fraction of the files needed to answer actually made it into the prompt
    # (graph mode also counts the files cited by its facts, since those facts are in the prompt too)
    # wiki mode has no passages; it counts the source files behind the pages it packed in
    seen = {cid.split("#")[0] for cid in trace["chunks"] + trace.get("sources", [])}
    for fact in trace.get("facts", []):
        seen.add(fact.rsplit("[", 1)[-1].rstrip("]").split("#")[0])
    return len(seen & set(gold_sources)) / len(gold_sources)


# ---------- running ----------

COLD_MS = 100  # a call that spent longer than this loading a model counts as cold


def warm_up(include_chat=True):
    # the first call loads each model into memory, which would make whichever mode runs first look slow
    print("Warming up models...")
    embed(["warm up"])
    if include_chat:
        ollama.chat(model=config.CHAT_MODEL, messages=[{"role": "user", "content": "Say ok."}],
                    options={"num_predict": 1}, keep_alive=config.KEEP_ALIVE)
        # if loading the chat model pushed the embed model out, every query will swap models; worth knowing up front
        stats = {}
        embed(["warm up"], stats=stats)
        if stats.get("load_ms", 0) > COLD_MS:
            print(f"  warning: the embed model reloaded after the chat model loaded ({stats['load_ms']:.0f} ms).\n"
                  "  Ollama can't keep both in memory, so expect cold calls. Try a smaller CHAT_MODEL or "
                  "set OLLAMA_MAX_LOADED_MODELS=2 before starting Ollama.")


def run_retrieval_only(question, index, mode, args):
    start = time.perf_counter()
    context, trace = build_context(question, index, mode, args.hops, args.top_entities, args.top_chunks,
                                   args.budget)
    total_ms = (time.perf_counter() - start) * 1000
    metrics = {
        "embed_ms": trace["embed_ms"],
        "retrieve_ms": trace["retrieve_ms"],
        "total_ms": total_ms,
        "warm_ms": total_ms - trace["embed_load_ms"],
        "embed_load_ms": trace["embed_load_ms"],
        "context_chars": len(context),
    }
    return {"answer": "", "trace": trace, "metrics": metrics}


def cmd_bench(args):
    run_bench(args)


def run_bench(args=None, **options):
    # notebooks call this with keyword options, e.g. run_bench(runs=1, retrieval_only=True)
    if args is None:
        args = default_args(**options)
    questions = json.loads(args.questions.read_text())
    if args.only:
        wanted = set(args.only.split(","))
        questions = [q for q in questions if q["id"] in wanted]
    modes = [m.strip() for m in args.modes.split(",")]
    index = load_index()

    warm_up(include_chat=not args.retrieval_only)

    rows = []
    total = args.runs * len(questions) * len(modes)
    for run in range(1, args.runs + 1):
        for position, q in enumerate(questions):
            # rotate the mode order by question and by run, so each mode takes each slot about equally often
            shift = (run + position) % len(modes)
            order = modes[shift:] + modes[:shift]
            for mode in order:
                print(f"  [{len(rows) + 1}/{total}] run {run}  {mode:6}  {q['id']}")
                if args.retrieval_only:
                    result = run_retrieval_only(q["question"], index, mode, args)
                else:
                    result = run_query(q["question"], index, mode, args.hops, args.top_entities,
                                       args.top_chunks, args.budget)
                metrics = result["metrics"]
                row = {"run": run, "mode": mode, "order": order.index(mode) + 1, "id": q["id"], "type": q["type"],
                       **metrics,
                       "cold": metrics.get("load_ms", 0) + metrics["embed_load_ms"] > COLD_MS,
                       "evidence_recall": evidence_recall(result["trace"], q["sources"]),
                       "facts": len(result["trace"].get("facts", [])),
                       "passages": len(result["trace"]["chunks"])}
                if not args.retrieval_only:
                    row["correct"] = is_correct(result["answer"], q["answers"])
                    row["answer"] = result["answer"]
                rows.append(row)

    summary = summarize(rows, modes, args.retrieval_only)
    print_report(summary, rows, questions, modes, args.retrieval_only)
    save(rows, summary, args)
    return rows, summary


# ---------- reporting ----------

def median(values):
    return statistics.median(values) if values else 0.0


def p95(values):
    # nearest-rank p95; with a handful of samples this is basically "the slow one"
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def summarize(rows, modes, retrieval_only):
    summary = {}
    for mode in modes:
        mine = [r for r in rows if r["mode"] == mode]
        s = {
            "n": len(mine),
            "cold_calls": sum(r["cold"] for r in mine),
            "warm_ms_median": median([r["warm_ms"] for r in mine]),
            "warm_ms_p95": p95([r["warm_ms"] for r in mine]),
            "total_ms_median": median([r["total_ms"] for r in mine]),
            "first_minus_second_ms": _order_effect(mine),
            "embed_ms_median": median([r["embed_ms"] for r in mine]),
            "retrieve_ms_median": median([r["retrieve_ms"] for r in mine]),
            "context_chars_median": median([r["context_chars"] for r in mine]),
            "evidence_recall": statistics.mean(r["evidence_recall"] for r in mine),
        }
        if not retrieval_only:
            correct = sum(r["correct"] for r in mine)
            prompt_tokens = sum(r["prompt_tokens"] for r in mine)
            gen_ms = sum(r["generate_ms"] for r in mine)
            s.update({
                "accuracy": correct / len(mine),
                "accuracy_simple": _acc(mine, "simple"),
                "accuracy_multi_hop": _acc(mine, "multi-hop"),
                "llm_ms_median": median([r["llm_ms"] for r in mine]),
                "prompt_tokens_median": median([r["prompt_tokens"] for r in mine]),
                "output_tokens_median": median([r["output_tokens"] for r in mine]),
                "generate_tok_per_s": sum(r["output_tokens"] for r in mine) / (gen_ms / 1000) if gen_ms else 0.0,
                "correct_per_1k_prompt_tokens": correct / prompt_tokens * 1000 if prompt_tokens else 0.0,
                "stable_questions": _stability(mine),
            })
        summary[mode] = s
    return summary


def _order_effect(rows):
    # first-minus-last warm latency, compared within each question (so a long answer to one question
    # can't masquerade as an order effect), then the median across questions. near 0 = order doesn't matter
    diffs = []
    last = max((r["order"] for r in rows), default=1)
    for qid in {r["id"] for r in rows}:
        first = [r["warm_ms"] for r in rows if r["id"] == qid and r["order"] == 1]
        second = [r["warm_ms"] for r in rows if r["id"] == qid and r["order"] == last]
        if first and second:
            diffs.append(median(first) - median(second))
    return median(diffs) if diffs else None


def _acc(rows, qtype):
    subset = [r for r in rows if r["type"] == qtype]
    return sum(r["correct"] for r in subset) / len(subset) if subset else None


def _stability(rows):
    # share of questions where every run agreed on right/wrong; temperature is 0 so this should be high
    by_q = {}
    for r in rows:
        by_q.setdefault(r["id"], set()).add(r["correct"])
    return sum(len(v) == 1 for v in by_q.values()) / len(by_q)


def print_report(summary, rows, questions, modes, retrieval_only):
    lines = [
        ("warm latency, median (ms)", "warm_ms_median", "{:.0f}"),
        ("warm latency, p95 (ms)", "warm_ms_p95", "{:.0f}"),
        ("raw latency incl. loads (ms)", "total_ms_median", "{:.0f}"),
        ("  question embed (ms)", "embed_ms_median", "{:.1f}"),
        ("  retrieval only (ms)", "retrieve_ms_median", "{:.2f}"),
    ]
    if not retrieval_only:
        lines += [
            ("  LLM answer (ms)", "llm_ms_median", "{:.0f}"),
            ("prompt tokens, median", "prompt_tokens_median", "{:.0f}"),
            ("output tokens, median", "output_tokens_median", "{:.0f}"),
            ("generation speed (tok/s)", "generate_tok_per_s", "{:.1f}"),
        ]
    lines += [
        ("context size, median (chars)", "context_chars_median", "{:.0f}"),
        ("evidence recall", "evidence_recall", "{:.0%}"),
    ]
    if not retrieval_only:
        lines += [
            ("accuracy, all", "accuracy", "{:.0%}"),
            ("accuracy, simple", "accuracy_simple", "{:.0%}"),
            ("accuracy, multi-hop", "accuracy_multi_hop", "{:.0%}"),
            ("correct per 1k prompt tokens", "correct_per_1k_prompt_tokens", "{:.2f}"),
            ("answer stability", "stable_questions", "{:.0%}"),
        ]

    lines += [
        ("cold calls (model loaded)", "cold_calls", "{:.0f}"),
        ("order effect, 1st - last (ms)", "first_minus_second_ms", "{:+.0f}"),
    ]
    width = 30
    print("\n" + "=" * (width + 14 * len(modes)))
    print(f"{'KPI':<{width}}" + "".join(f"{m:>14}" for m in modes))
    print("-" * (width + 14 * len(modes)))
    for label, key, fmt in lines:
        cells = "".join(f"{fmt.format(summary[m][key]) if summary[m][key] is not None else '-':>14}" for m in modes)
        print(f"{label:<{width}}{cells}")
    print("=" * (width + 14 * len(modes)))

    if not retrieval_only:
        print("\nPer question (runs correct / total):")
        for q in questions:
            cells = []
            for mode in modes:
                mine = [r for r in rows if r["id"] == q["id"] and r["mode"] == mode]
                cells.append(f"{mode} {sum(r['correct'] for r in mine)}/{len(mine)}")
            print(f"  {q['id']:<16} {q['type']:<10} " + "   ".join(cells))


def save(rows, summary, args):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S") + ("-retrieval" if args.retrieval_only else "")
    n = 2
    while (RESULTS_DIR / f"bench-{stamp}.json").exists():  # two runs in the same second shouldn't overwrite
        stamp = f"{stamp.split('_')[0]}_{n}"
        n += 1
    meta = load_meta()  # how the index was built, not the current env, since that's what was benchmarked
    run_config = {"chat_model": config.CHAT_MODEL, "embed_model": config.EMBED_MODEL, "budget": args.budget,
              "index_dir": str(config.INDEX_DIR), "chunk_method": meta.get("chunk_method"),
              "chunk_count": meta.get("chunk_count"), "chunk_overlap": meta.get("chunk_overlap"),
              "chunk_chars": meta.get("chunk_chars", config.CHUNK_CHARS), "runs": args.runs, "hops": args.hops,
              "top_entities": args.top_entities, "top_chunks": args.top_chunks,
              "retrieval_only": args.retrieval_only}
    json_path = RESULTS_DIR / f"bench-{stamp}.json"
    json_path.write_text(json.dumps({"config": run_config, "summary": summary, "rows": rows}, indent=1))

    csv_path = RESULTS_DIR / f"bench-{stamp}.csv"
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {os.path.relpath(json_path)} and {csv_path.name}")
