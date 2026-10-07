# shared fixtures for the test suite
#
# nothing here talks to a real Ollama: FakeOllama stands in for both models, so the whole suite runs
# in a few seconds with no models pulled. tests also run in a temp folder with their own copy of the
# sample docs, so they never touch the real index/ or wiki/ and don't break if data/ changes.
import hashlib
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
import ollama  # noqa: E402
import okf  # noqa: E402

# a frozen copy of the sample corpus
DOCS = {
    "incidents.md": """# Incident log, spring quarter

On March 14, the payments API went down for three hours. The root cause was a memory leak in the Redwood caching library, which the payments API uses to store session data.

On April 2, the shipment tracking dashboard showed stale data for twenty minutes after a misconfigured deploy. No third-party library was at fault.

After the March outage, Dana Okafor asked every team to review its third-party dependencies.
""",
    "team.md": """# Engineering teams at Northwind Freight

Priya Raman leads the Atlas team. Atlas owns the payments API, which handles every customer invoice and refund.

Marcus Chen leads the Beacon team. Beacon builds the shipment tracking dashboard used by warehouse staff.

Both team leads report to Dana Okafor, the VP of Engineering, who joined Northwind Freight in 2023.
""",
    "vendors.md": """# Third-party vendors

Redwood is a caching library maintained by Juniper Labs, a small vendor based in Portland, Oregon.

The shipment tracking dashboard uses Brightline Charts, a charting library made by Halcyon Software.

Juniper Labs released a patch for the Redwood memory leak on March 20.
""",
}

NAME = re.compile(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)*")


def fake_vector(text, dims=256):
    # bag-of-words hashed into a vector: texts that share words come out similar, which is all retrieval needs
    vector = np.zeros(dims)
    for word in re.findall(r"\w+", text.lower()):
        vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % dims] += 1
    return (vector + 1e-3).tolist()


class FakeOllama:
    def __init__(self):
        self.chat_calls, self.embed_calls = [], []
        self.queued = []          # canned chat replies, used before the default behavior
        self.chat_load_ns = 0     # pretend model-load times, for the warm-latency math
        self.embed_load_ns = 0

    def chat(self, model, messages, format=None, options=None, keep_alive=None):
        prompt = messages[0]["content"]
        self.chat_calls.append({"model": model, "prompt": prompt, "format": format})
        if self.queued:
            content = self.queued.pop(0)
        elif format == "json":
            content = json.dumps(self.extract(prompt.split("Text:\n", 1)[-1]))
        else:
            # answers "Juniper Labs" only when that name made it into the context
            content = "Juniper Labs [vendors#0]" if "Juniper Labs" in prompt else "I don't know."
        return {"message": {"content": content}, "prompt_eval_count": len(prompt) // 4, "eval_count": 7,
                "prompt_eval_duration": 3_000_000, "eval_duration": 5_000_000, "load_duration": self.chat_load_ns}

    @staticmethod
    def extract(text):
        # crude stand-in for the LLM: capitalized phrases are entities, neighbors get a relation
        names = list(dict.fromkeys(NAME.findall(text)))
        entities = [{"name": n, "type": "other", "description": f"mentioned near {text[:20]!r}"} for n in names]
        relations = [{"source": a, "relation": "mentioned with", "target": b} for a, b in zip(names, names[1:])]
        return {"entities": entities, "relations": relations}

    def embed(self, model, input, keep_alive=None):
        self.embed_calls.append(list(input))
        return {"embeddings": [fake_vector(t) for t in input], "load_duration": self.embed_load_ns}


@pytest.fixture
def fake_ollama(monkeypatch):
    fake = FakeOllama()
    monkeypatch.setattr(ollama, "chat", fake.chat)
    monkeypatch.setattr(ollama, "embed", fake.embed)
    return fake


@pytest.fixture
def project(tmp_path, monkeypatch, fake_ollama):
    # a throwaway copy of the project layout; monkeypatch puts every setting back after the test
    data = tmp_path / "data"
    data.mkdir()
    for name, text in DOCS.items():
        (data / name).write_text(text, encoding="utf-8")
    settings = {"DATA_DIR": data, "INDEX_DIR": tmp_path / "index", "WIKI_DIR": tmp_path / "wiki",
                "CHUNK_METHOD": "paragraph", "CHUNK_CHARS": 200, "CHUNK_OVERLAP": 50,
                "CHAT_MODEL": "fake-chat", "EMBED_MODEL": "fake-embed"}
    for key, value in settings.items():
        monkeypatch.setattr(config, key, value)
    monkeypatch.chdir(tmp_path)
    okf._cache.clear()
    return SimpleNamespace(root=tmp_path, data=data, fake=fake_ollama)


@pytest.fixture
def built(project, capsys):
    # the project after a full `index` run
    import graphrag
    graphrag.cmd_index(SimpleNamespace())
    capsys.readouterr()  # swallow the progress output
    project.index = graphrag.load_index()
    return project
