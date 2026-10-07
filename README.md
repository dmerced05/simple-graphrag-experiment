# Simple GraphRAG Experiment

A local, hands-on playground for comparing ways of doing retrieval-augmented generation (RAG): plain vector search, a knowledge graph, and a compiled Markdown wiki in the Open Knowledge Format (OKF). Everything runs on your own machine with Ollama models.

## How to run

### 1. Install

1. **Ollama:** install it from [ollama.com](https://ollama.com) and open it, so it runs in the background.
2. **Models:** pull the chat model and the embedding model:
   ```bash
   ollama pull qwen2.5:7b
   ollama pull nomic-embed-text
   ```
   `qwen2.5:7b` needs about 8 GB of free RAM. On a smaller machine, pull `qwen2.5:3b` and run `export CHAT_MODEL=qwen2.5:3b` (Windows PowerShell: `$env:CHAT_MODEL="qwen2.5:3b"`). Any chat model that can output JSON works.
3. **Python packages** (Python 3.10+), from the project folder:
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   pip install ipykernel pandas matplotlib   # only for the notebook in testing/
   ```

### 2. Build the index and ask questions

```bash
python graphrag.py index      # chunk, extract, build the graph, embed, write the wiki (about 1-2 min)
python graphrag.py stats      # what got extracted

python graphrag.py ask "Who leads the Beacon team?" --mode vector
python graphrag.py ask "Who leads the Beacon team?" --mode graph
python graphrag.py ask "Who leads the Beacon team?" --mode wiki
```

Add `--show-context` to see exactly what was retrieved, and `--timing` for latency and token counts. Run `ask` with no question for interactive mode. Extractions are cached in `index/extractions.json`, so re-running `index` only processes new or changed chunks.

The repo includes a prebuilt `index/` and `wiki/`, so you can run `ask` right away and rebuild later.

### 3. Compare the three modes

```bash
python graphrag.py bench --retrieval-only    # fast: retrieval speed and evidence recall
python graphrag.py bench                     # full: accuracy, latency, tokens (about 10-15 min)
```

### 4. Run the tests

```bash
python -m pytest testing     # about a second; no Ollama needed
```

For a guided walkthrough, open `testing/bench.ipynb` and run it section by section. If something goes wrong, see [Troubleshooting](#troubleshooting).

## About this project

This is a learning project: a hands-on way to see how RAG systems actually behave by building the pieces myself and measuring them, instead of treating a framework as a black box. It runs entirely on local Ollama models, so every step can be inspected: which chunks were created, what the model extracted, what got retrieved, and how many tokens and milliseconds each answer cost.

The same questions are answered three ways:

| Mode | What goes into the prompt |
| --- | --- |
| `vector` | The 3 chunks most similar to the question (the plain RAG baseline) |
| `graph` | Facts from walking a knowledge graph of extracted entities and relations, plus the source chunks they came from |
| `wiki` | Pages from a Markdown wiki compiled from the graph in the [Open Knowledge Format](https://github.com/GoogleCloudPlatform/knowledge-catalog/tree/main/okf), packed into a fixed token budget, with no raw text |

The sample corpus is tiny (three short files about a fictional company), so the results are directional, not conclusions.

## Project files

Each step of the pipeline lives in its own file. `graphrag.py` is the command-line entry point that runs them in order.

```
graphrag.py            # command line: index, chunks, ask, stats, export, export-okf, bench

# index time (python graphrag.py index runs these in order)
chunking.py            # 1. load documents from data/ and split them into chunks (3 methods)
extraction.py          # 2. LLM extracts entities + relations from each chunk (cached)
knowledge_graph.py     # 3. merge extractions into one graph; save/load it as JSON
embedding.py           # 4. embed entities and chunks (also used for every question)
index_store.py         # 5. save/load everything in index/
okf.py                 # 6. compile the graph into the OKF wiki (+ wiki-mode retrieval)

# query time (python graphrag.py ask runs these)
retrieval.py           # 7. build the context for vector, graph or wiki mode
generation.py          # 8. ask the chat model, record timings and token counts

config.py              # settings (models, folders, chunk size), all overridable with env vars

testing/
  bench.py             # benchmark: graph vs. vector vs. wiki (run with `python graphrag.py bench`)
  bench.ipynb          # feature-test notebook, one section per feature
  tests/               # pytest suite (fake Ollama, runs in about a second)
  benchmark/questions.json   # benchmark questions with expected answers
  benchmark/results/   # one CSV + JSON per benchmark run

data/                  # sample documents (a fictional company)
index/                 # built by `index`: graph, chunks, embeddings, extraction cache
wiki/                  # built by `index` / `export-okf`: the OKF wiki
```

To try a different approach to one step, like a new chunking strategy, edit that step's file. Everything else should stay the same. (The chunking file is `chunking.py`, not `chunk.py`, because Python's standard library already has a `chunk` module and Jupyter can load that one instead.)

## Chunking methods

How documents get split affects everything after it: what the LLM extracts, what vector mode retrieves, and what the wiki contains. There are three methods:

| Method | What it does | Trade-off |
| --- | --- | --- |
| `paragraph` (default) | Packs whole paragraphs into chunks up to `CHUNK_CHARS` | Headings can end up as tiny chunks of their own, which produce junk entities |
| `fixed` | Fixed windows of `CHUNK_CHARS`, each repeating the last `CHUNK_OVERLAP` characters of the previous one. Cut points snap to spaces so words stay whole. | The common baseline approach. Facts near a boundary survive because of the overlap, but sentences get cut and the overlap gets extracted twice. |
| `structure` | Splits by Markdown headings, then paragraphs, then sentences if a paragraph is too long. Every chunk starts with its heading path, like `Third-party vendors`. | Chunks carry their section's context and headings never stand alone. Depends on documents having structure. The heading path isn't counted toward `CHUNK_CHARS`. |

Preview a method without calling the LLM:

```bash
python graphrag.py chunks --chunk-method structure --full
python graphrag.py chunks --chunk-method fixed --chunk-chars 300 --chunk-overlap 60
```

To compare methods, give each one its own index and wiki folder, so they don't overwrite each other:

```bash
INDEX_DIR=index_structure WIKI_DIR=wiki_structure python graphrag.py index --chunk-method structure
INDEX_DIR=index_structure WIKI_DIR=wiki_structure python graphrag.py bench
```

(Windows PowerShell: set `$env:INDEX_DIR="index_structure"` and `$env:WIKI_DIR="wiki_structure"` first.)

Each index saves how it was built in `index/meta.json`. Benchmark results record the chunking method, so the notebook's saved-run comparison can tell them apart. Section 10 of `testing/bench.ipynb` builds and benchmarks all three methods in one go.

Changing the method or size changes the chunk text, so the first `index` with a new setting re-extracts. After that it's cached. The `paragraph` method's output is pinned by a test, so existing indexes and extraction caches stay valid.

## The demo multi-hop questions

The sample docs in `data/` describe a fictional company, with facts deliberately spread across three files. This question needs four facts from all three files (Priya leads Atlas → Atlas owns the payments API → the outage was caused by Redwood → Redwood is maintained by Juniper Labs):

```bash
python graphrag.py ask "Which vendor maintains the library that caused the outage in the system Priya Raman's team owns?" --mode vector --show-context
python graphrag.py ask "Which vendor maintains the library that caused the outage in the system Priya Raman's team owns?" --mode graph --show-context
python graphrag.py ask "Which vendor maintains the library that caused the outage in the system Priya Raman's team owns?" --mode wiki --show-context
```

Vector mode grabs the passages that sound most like the question and can miss a file. Graph and wiki mode follow the connections.

More questions to try (all are in `testing/benchmark/questions.json`):

| Question | Answer | Type |
| --- | --- | --- |
| Who leads the Beacon team? | Marcus Chen | Simple |
| Who does the leader of the team that owns the payments API report to? | Dana Okafor | Multi-hop |
| When was the fix released for the problem that took down the payments API? | March 20 | Multi-hop |
| Which company makes the charting library used by Marcus Chen's team's dashboard? | Halcyon Software | Multi-hop |

Results vary between runs and models, since small models extract imperfectly.

## The OKF wiki

`index` also compiles the graph into a wiki in the [Open Knowledge Format](https://github.com/GoogleCloudPlatform/knowledge-catalog/tree/main/okf) (OKF v0.1). OKF is plain `.md` files with YAML frontmatter, not a special file type, so the wiki opens in any editor. In VS Code, right-click a page and choose **Open Preview** to follow the links.

```bash
python graphrag.py export-okf                            # rebuild wiki/ from the existing index (no LLM calls)
python graphrag.py ask "..." --mode wiki --show-context  # see which pages were packed in
python graphrag.py ask "..." --mode wiki --budget 150    # squeeze the token budget
```

```
wiki/
├── index.md                  # top-level overview, links to each folder
├── log.md                    # one entry per compile, plus possible duplicate entities
├── person/index.md
├── person/priya-raman.md
└── system/payments-api.md ...
```

Each page has `type` (the only field OKF requires), `title`, `description`, `resource`, `tags` and `timestamp`. It also has custom fields: `id` (the graph node), `sources` (chunk ids, for provenance), `degree` and `token_footprint`. Relations are standard Markdown links with a source citation, e.g. `- **Atlas team** owns → [payments API](/system/payments-api.md) [team#0]`.

The wiki is generated from `index/graph.json`, not a replacement for it. It's rebuilt from scratch on every `index` or `export-okf`, so hand edits to pages will be overwritten (`log.md` is kept).

`log.md` lists entity pairs that look like duplicates, such as "Redwood" and "Redwood caching library". They aren't merged automatically. They matter most in wiki mode, which has no raw text to fall back on.

## Benchmark

`bench` runs every question in `testing/benchmark/questions.json` through each mode and prints a side-by-side KPI table:

```bash
python graphrag.py bench                             # all three modes, 3 runs per question per mode
python graphrag.py bench --runs 5                    # more runs = steadier timings
python graphrag.py bench --retrieval-only            # skip the LLM and time retrieval alone (fast)
python graphrag.py bench --modes graph,vector        # pick which modes to compare
python graphrag.py bench --modes wiki --budget 150   # one mode, smaller budget
python graphrag.py bench --only outage-vendor,fix-date
```

Each run saves its raw rows to `testing/benchmark/results/bench-<timestamp>.csv` and `.json`, along with the model and settings used.

It controls for run order in three ways:

- **Warm-up:** it loads both models before timing starts.
- **Keep-alive:** it keeps both models loaded with `KEEP_ALIVE` (default `30m`).
- **Rotation:** it rotates which mode goes first by question and by run, so each mode takes each slot equally often.

Any model load that still happens is measured and subtracted from warm latency.

| KPI | What it tells you |
| --- | --- |
| Warm latency, median and p95 | End-to-end time per question with model loading subtracted. This is the fair comparison. p95 shows the slow cases. |
| Raw latency | The same, but including any model loads. |
| Question embed / retrieval / LLM answer | Where the time goes. Retrieval is the only step where the modes do different work. |
| Prompt tokens | How much context each mode sends to the model. This is a proxy for cost. |
| Output tokens, generation speed | Answer length and model throughput. Output tokens are the slow part locally. |
| Context size (chars) | Prompt size without relying on Ollama's token count. |
| Evidence recall | Share of the source files needed for the answer that made it into the prompt. |
| Accuracy (all / simple / multi-hop) | Whether the answer mentions the expected answer. |
| Correct per 1k prompt tokens | Accuracy per unit of context: how much useful answer each token of context buys. |
| Answer stability | Share of questions where every run agreed on right or wrong. |
| Cold calls | Queries where Ollama had to load a model first. Should be 0. If not, Ollama is swapping models because of memory. |
| Order effect | Warm latency when a mode ran first minus when it ran last, compared within each question. Near 0 means run order isn't skewing results. |

To add questions, append to `testing/benchmark/questions.json` with the accepted `answers` and the `sources` (file names without extension) the answer depends on. Accuracy is a substring match, so an answer like "not Juniper Labs" would still count as correct.

Two caveats when reading results:

- **Prompt cache:** Ollama's prompt cache can make prompt reading much faster when consecutive prompts start the same way. This mostly helps graph mode. Token counts aren't affected.
- **Tiny corpus:** with only 9 chunks, results are directional.

## Testing

Everything test-related lives in `testing/`.

### Unit tests (pytest)

```bash
python -m pytest testing          # from the project folder; about a second
python -m pytest testing -k okf   # just the wiki tests
```

The suite doesn't need Ollama. A fake Ollama stands in for both models, and every test runs in a temporary folder with a frozen copy of the sample docs, so it never touches your real `index/` or `wiki/`. There's one test file per pipeline step:

| File | Covers |
| --- | --- |
| `test_chunking.py` | All three methods: chunk shape and ids, exact boundaries, overlap, no chopped words, heading paths |
| `test_extraction.py` | JSON parsing and filtering, the single retry, caching, re-extraction on a model change |
| `test_knowledge_graph.py` | Name normalization, merging, self-loops, chunk provenance, JSON round trip |
| `test_embedding_and_index.py` | Unit vectors, batching, load-time stats, index save/load, `meta.json` |
| `test_okf.py` | One page per entity, required `type`, links resolve, log kept, slug collisions, duplicate lint, budget, named pages first, no repeated lines |
| `test_retrieval_generation.py` | What each mode sends, citations, prompt shape, metrics and warm-latency math |
| `test_bench.py` | Scoring, recall, p95, order effect, balanced mode rotation, saving, the real questions file |
| `test_cli.py` | Every `graphrag.py` command end to end |

Run it after any code change. It's the quickest way to know you didn't break another step.

### Feature notebook

Open `testing/bench.ipynb`, pick your Python environment as the kernel, and run **Setup** first. It switches to the project folder on its own. Each section tests one feature and is labeled with whether it needs Ollama:

1. Unit tests (runs pytest)
2. Chunking methods, side by side
3. Index and extraction, with what each chunk produced
4. Knowledge graph
5. OKF wiki checks (required fields, broken links, duplicate log)
6. Retrieval per mode, showing exactly what each mode sends
7. Answering one question in all three modes
8. Benchmark (retrieval-only and full) with charts
9. Wiki token budget sweep
10. Chunking method comparison
11. Saved benchmark runs

Restart the kernel after pulling new code. You can also call things directly in a cell:

```python
import graphrag
from testing import bench
index = graphrag.load_index()
result = graphrag.run_query("Who leads the Beacon team?", index, "wiki", budget=300)
rows, summary = bench.run_bench(modes="wiki", runs=1, budget=150)
```

## How it works

The file for each step is in parentheses.

**Index:**

1. Load files and split them into chunks (`chunking.py`).
2. Ask the LLM to extract entities and relations from each chunk as JSON (`extraction.py`).
3. Merge duplicate entity names, so "The Payments API" and "payments api" become one node, and build a NetworkX graph where every node and edge remembers which chunk it came from (`knowledge_graph.py`).
4. Embed entity descriptions and chunks (`embedding.py`).
5. Save everything to `index/` (`index_store.py`).
6. Write the OKF wiki to `wiki/` (`okf.py`).

**Answering** happens in `retrieval.py`, which builds the context for each mode, then `generation.py`, which asks the chat model.

**Vector mode:** embed the question → take the most similar chunks → give them to the LLM. That's the baseline.

**Graph mode:** embed the question → pick the most similar entities as entry points, plus any entity named in the question → walk 2 hops outward → turn every relation in that neighborhood into a readable fact → pull the source chunks attached to those entities → give the facts and passages to the LLM.

**Wiki mode:**

1. Pick seed pages: the most similar pages, plus any page the question names outright.
2. Follow the wiki's links up to `--hops` away.
3. Rank pages by similarity, minus a penalty for each hop away from a seed. Pages the question names always rank first.
4. Pack pages greedily into `--budget` tokens (estimated as characters / 4), skipping relation lines an earlier page already added.
5. Give only those compact pages to the LLM.

This treats retrieval as a small budgeting problem: get the most useful context into a fixed number of tokens, without paying twice for the same fact.

## Options

| Flag or variable | Default | What it does |
| --- | --- | --- |
| `--mode` | `graph` | `graph`, `vector` or `wiki` |
| `--hops` | `2` | How far to walk from the entry points (graph and wiki) |
| `--top-entities` | `5` | How many entry-point entities to start from |
| `--top-chunks` | `3` | How many source passages go into the prompt (graph and vector) |
| `--budget` | `300` | Wiki mode: max context tokens |
| `--show-context` | off | Print what was retrieved |
| `--timing` | off | Print latency and token counts after the answer |
| `CHAT_MODEL` | `qwen2.5:7b` | Model for extraction and answers |
| `EMBED_MODEL` | `nomic-embed-text` | Model for embeddings |
| `KEEP_ALIVE` | `30m` | How long Ollama keeps models loaded between calls |
| `CHUNK_METHOD` / `--chunk-method` | `paragraph` | `paragraph`, `fixed` or `structure` |
| `CHUNK_CHARS` / `--chunk-chars` | `200` | Max characters per chunk |
| `CHUNK_OVERLAP` / `--chunk-overlap` | `50` | `fixed` method only: characters repeated between neighboring chunks |
| `DATA_DIR` / `INDEX_DIR` / `WIKI_DIR` | `data` / `index` / `wiki` | Where docs are read from and outputs are saved |

`python graphrag.py export` writes `index/graph.graphml`, which you can open in [Gephi](https://gephi.org) or yEd to see the graph.

## Using your own documents

Drop `.md` or `.txt` files into `data/` (or point `DATA_DIR` somewhere else) and run `index` again. For real documents, raise the chunk size: `export CHUNK_CHARS=1000`. The default of 200 is small on purpose so the tiny sample docs split into several chunks. Update `testing/benchmark/questions.json` with questions about your documents before benchmarking.

PDFs aren't supported yet. Convert them to text first.

## Troubleshooting

- **"Can't reach Ollama"**: the Ollama app isn't running. Open it, or run `ollama serve`.
- **"model not found"**: you haven't pulled that model yet. Run the `ollama pull` commands above.
- **"No wiki in wiki/"**: run `python graphrag.py export-okf`. In the notebook, also check that `os.getcwd()` is the project folder.
- **`No module named 'yaml'`**: run `pip install -r requirements.txt` again (it includes `pyyaml`).
- **Notebook doesn't see new code**: restart the kernel.
- **`No module named 'testing'`** when running the notebook: run the Setup cell first; it switches to the project folder.
- **Cold calls aren't 0, or the warm-up warns that the embed model reloaded**: your machine can't hold both models at once. Use a smaller `CHAT_MODEL`, or set `OLLAMA_MAX_LOADED_MODELS=2` before starting Ollama.
- **Graph has very few relations**: the chat model is struggling with extraction. Try a larger model.
- **Changed models and want fresh extractions**: the cache is keyed by model name, so this happens automatically. To force a full rebuild, delete the `index/` folder.
