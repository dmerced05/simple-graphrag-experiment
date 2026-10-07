# step 4: turn text into vectors with the embedding model (used at index time and for every question)
import numpy as np
import ollama

import config


def ns_to_ms(value):
    # Ollama reports durations in nanoseconds
    return value / 1e6 if value else 0.0


def embed(texts, batch_size=32, stats=None):
    vectors = []
    for i in range(0, len(texts), batch_size):
        response = ollama.embed(model=config.EMBED_MODEL, input=texts[i:i + batch_size], keep_alive=config.KEEP_ALIVE)
        vectors.extend(response["embeddings"])
        if stats is not None:  # lets the benchmark see if this call had to load the model first
            stats["load_ms"] = stats.get("load_ms", 0.0) + ns_to_ms(response.get("load_duration"))
    vectors = np.array(vectors, dtype=np.float32)
    # unit length means cosine similarity is just a dot product later
    return vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)


def entity_text(data):
    # what I embed for each entity: its name, type and everything extraction said about it
    return f"{data['name']} ({data['type']}): {' '.join(data['descriptions'])}".strip()
