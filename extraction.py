# step 2: ask the LLM for entities and relations in each chunk (one call per chunk, cached so reruns resume)
import hashlib
import json

import ollama

import config

EXTRACT_PROMPT = """Extract the entities and relationships from the text below.
Return JSON only, in exactly this shape:
{"entities": [{"name": "...", "type": "person|organization|team|product|system|event|place|other", "description": "one short sentence"}],
 "relations": [{"source": "entity name", "relation": "short verb phrase", "target": "entity name"}]}
Rules: only use facts stated in the text. Use names exactly as written. Every relation's source and target must also appear in entities.

Text:
<<CHUNK>>"""


def extract(chunk_text):
    prompt = EXTRACT_PROMPT.replace("<<CHUNK>>", chunk_text)
    for _ in range(2):  # small models occasionally return junk, so I give it one retry
        try:
            response = ollama.chat(
                model=config.CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options={"temperature": 0},
            )
            parsed = json.loads(response["message"]["content"])
            entities = [e for e in parsed.get("entities", []) if isinstance(e, dict) and e.get("name")]
            relations = [r for r in parsed.get("relations", []) if isinstance(r, dict) and r.get("source") and r.get("target")]
            return {"entities": entities, "relations": relations}
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return {"entities": [], "relations": []}


def extract_all(chunks):
    cache_path = config.INDEX_DIR / "extractions.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    results = {}
    for i, chunk in enumerate(chunks, 1):
        # keyed on model + text, so changing either re-extracts just that chunk
        cache_key = hashlib.sha1(f"{config.CHAT_MODEL}|{chunk['text']}".encode()).hexdigest()
        if cache_key not in cache:
            print(f"  extracting {i}/{len(chunks)}  {chunk['id']}")
            cache[cache_key] = extract(chunk["text"])
            cache_path.write_text(json.dumps(cache))  # save after every chunk so a crash doesn't lose work
        else:
            print(f"  cached     {i}/{len(chunks)}  {chunk['id']}")
        results[chunk["id"]] = cache[cache_key]
    return results
