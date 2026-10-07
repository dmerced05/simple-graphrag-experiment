# step 1: load documents from DATA_DIR and split them into chunks
#
# three methods, picked with CHUNK_METHOD (or --chunk-method):
#   paragraph  (default) glue whole paragraphs together up to CHUNK_CHARS. the original behavior
#   fixed      level 1: fixed-size windows of CHUNK_CHARS that overlap by CHUNK_OVERLAP characters
#   structure  level 2: split by markdown headings, then paragraphs, then sentences;
#              every chunk starts with its heading path so it keeps its section's context
#
# sizes are in characters (~4 chars per token), to match the rest of the project
import re
import sys

import config

METHODS = ("paragraph", "fixed", "structure")
HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def load_chunks(method=None):
    method = method or config.CHUNK_METHOD
    if method not in METHODS:
        sys.exit(f"Unknown CHUNK_METHOD '{method}'. Pick one of: {', '.join(METHODS)}")
    files = sorted(list(config.DATA_DIR.glob("*.md")) + list(config.DATA_DIR.glob("*.txt")))
    if not files:
        sys.exit(f"No .md or .txt files found in {config.DATA_DIR}/")

    splitter = {"paragraph": split_paragraphs, "fixed": split_fixed, "structure": split_structure}[method]
    chunks = []
    for file in files:
        pieces = splitter(file.read_text(encoding="utf-8"))
        # tag each chunk with its source file so answers can cite it
        for n, text in enumerate(pieces):
            chunks.append({"id": f"{file.stem}#{n}", "source": file.name, "text": text, "method": method})
    return chunks


# ---------- shared helper ----------

def pack(pieces, limit, joiner="\n\n"):
    # glue pieces together until adding the next one would pass the limit.
    # a single piece bigger than the limit becomes its own chunk (callers split those further if they care)
    packed, current = [], ""
    for piece in pieces:
        if current and len(current) + len(joiner) + len(piece) > limit:
            packed.append(current)
            current = ""
        current = f"{current}{joiner}{piece}" if current else piece
    if current:
        packed.append(current)
    return packed


def paragraphs_of(text):
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


# ---------- original: whole paragraphs ----------

def split_paragraphs(text):
    # I glue paragraphs together until a chunk gets big, so short paragraphs keep their neighbors as context.
    # kept byte-for-byte as the original loop so existing indexes and extraction caches stay valid
    pieces, current = [], ""
    for paragraph in paragraphs_of(text):
        if current and len(current) + len(paragraph) > config.CHUNK_CHARS:
            pieces.append(current)
            current = ""
        current = f"{current}\n\n{paragraph}".strip()
    if current:
        pieces.append(current)
    return pieces


# ---------- level 1: fixed windows with overlap ----------

def split_fixed(text):
    # ignores structure on purpose; this is the "industry baseline" (GraphRAG uses 1200 tokens + overlap).
    # I only nudge the cut points to whitespace so words don't get chopped in half
    text = re.sub(r"\s+", " ", text).strip()
    size = config.CHUNK_CHARS
    overlap = min(config.CHUNK_OVERLAP, size // 2)  # overlap >= size would never move forward
    pieces, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            space = text.rfind(" ", start + size // 2, end)
            if space != -1:
                end = space
        pieces.append(text[start:end].strip())
        if end >= len(text):
            break
        # next window starts `overlap` characters back, at the start of a word
        next_start = end - overlap
        space = text.find(" ", next_start, end)
        next_start = space + 1 if space != -1 else next_start
        start = max(next_start, start + 1)  # always make progress
    return [p for p in pieces if p]


# ---------- level 2: structure-aware ----------

def sections_of(text):
    # walk the lines, tracking the current heading path, e.g. ["Incident log, spring quarter"]
    sections, path, body = [], [], []
    for line in text.splitlines():
        match = HEADING.match(line)
        if match:
            if "".join(body).strip():
                sections.append((list(path), "\n".join(body)))
            body = []
            level = len(match.group(1))
            path = path[:level - 1] + [match.group(2)]
        else:
            body.append(line)
    if "".join(body).strip():
        sections.append((list(path), "\n".join(body)))
    return sections  # headings with no text under them don't become chunks


def split_structure(text):
    limit = config.CHUNK_CHARS
    chunks = []
    for path, body in sections_of(text):
        prefix = " > ".join(path)
        pieces = []
        for paragraph in paragraphs_of(body):
            if len(paragraph) <= limit:
                pieces.append(paragraph)
            else:
                # paragraph too big on its own: fall back to sentences, packed back up to the limit
                pieces.extend(pack(SENTENCE_END.split(paragraph), limit, joiner=" "))
        for piece in pack(pieces, limit):
            # the heading path rides along so the chunk still says what section it's from
            chunks.append(f"{prefix}\n\n{piece}" if prefix else piece)
    return chunks
