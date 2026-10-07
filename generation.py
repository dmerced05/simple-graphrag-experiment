# step 8 (query time): retrieve context, ask the chat model, and record timings + token counts
import time

import ollama

import config
from embedding import ns_to_ms
from retrieval import build_context

ANSWER_PROMPT = """Answer the question using only the facts and passages below.
If the answer takes several steps, chain the facts together and show the chain briefly.
If the answer isn't there, say you don't know. Cite passage ids in [brackets].

<<CONTEXT>>

Question: <<QUESTION>>"""


def run_query(question, index, mode, hops=2, top_entities=5, top_chunks=3, budget=300):
    # one full question -> answer pass, with timings and token counts so bench.py can compare modes
    start = time.perf_counter()
    context, trace = build_context(question, index, mode, hops, top_entities, top_chunks, budget)
    prompt = ANSWER_PROMPT.replace("<<CONTEXT>>", context).replace("<<QUESTION>>", question)

    llm_start = time.perf_counter()
    response = ollama.chat(model=config.CHAT_MODEL, messages=[{"role": "user", "content": prompt}],
                           options={"temperature": 0}, keep_alive=config.KEEP_ALIVE)
    llm_ms = (time.perf_counter() - llm_start) * 1000

    # Ollama reports its own counters in nanoseconds; I keep them next to my wall-clock numbers
    load_ms = ns_to_ms(response.get("load_duration"))
    total_ms = (time.perf_counter() - start) * 1000
    metrics = {
        "embed_ms": trace["embed_ms"],
        "retrieve_ms": trace["retrieve_ms"],
        "llm_ms": llm_ms,
        "total_ms": total_ms,
        # warm_ms = what the query costs once both models are already in memory; this is the fair number to compare
        "warm_ms": total_ms - load_ms - trace["embed_load_ms"],
        "load_ms": load_ms,
        "embed_load_ms": trace["embed_load_ms"],
        "prompt_eval_ms": ns_to_ms(response.get("prompt_eval_duration")),
        "generate_ms": ns_to_ms(response.get("eval_duration")),
        "prompt_tokens": response.get("prompt_eval_count") or 0,
        "output_tokens": response.get("eval_count") or 0,
        "context_chars": len(context),
    }
    return {"answer": response["message"]["content"].strip(), "trace": trace, "metrics": metrics}
