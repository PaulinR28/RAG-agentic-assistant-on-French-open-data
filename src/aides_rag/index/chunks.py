"""Turn the aid corpus into retrieval chunks (one aid = one chunk, long aids split by section).

Each chunk = a metadata header (title, aid type, area, funders, deadline...) + body text.
The header is repeated on every chunk of an aid, so each chunk is self-contained for
BM25, the embedding model and the reranker. Template placeholders (%pme%, %mini%) are
expanded into their meaning.

Sizes are in characters; for our French text, 1 token ~ 4.0 characters (measured with the bge-m3 / XLM-R
tokenizer: mean 3.98, 5th percentile 3.66).
The default body budget of 1,800 characters (~500 tokens) keeps header + body inside a
512-token window for 87% of chunks (max 636 real tokens, well inside bge-m3's 1,024 setting);
long sections are split on sentences with ~15% overlap.

Usage:
    uv run python -m aides_rag.index.chunks --corpus data/processed/aids.jsonl --out data/processed/chunks.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

MAX_BODY_CHARS = 1800
OVERLAP = 0.15
CHARS_PER_TOKEN = 4.0  # measured, see docstring; tests/test_hybrid.py checks real token counts
MAX_LABELS = 5  # header lists (communes, funders) are capped: one aid lists 613 communes
MAX_LABEL_CHARS = 250

SECTIONS = [
    ("objet", "Objet"),
    ("operations", "Dépenses et opérations éligibles"),
    ("conditions", "Conditions"),
    ("montant", "Montant"),
    ("beneficiaires", "Bénéficiaires"),
]

PLACEHOLDERS = {
    "%pme%": (
        "PME au sens de la définition européenne (moins de 250 salariés, chiffre d'affaires "
        "annuel inférieur ou égal à 50 M€ ou total de bilan inférieur ou égal à 43 M€)"
    ),
    "%mini%": (
        "Aide soumise au règlement de minimis : le total des aides de minimis reçues par "
        "l'entreprise est plafonné sur trois exercices fiscaux."
    ),
}
PLACEHOLDER_RE = re.compile("|".join(map(re.escape, PLACEHOLDERS)))
SENTENCE_RE = re.compile(r"(?<=[.;:!?])\s+|\s+(?=- )")

SCOPE_LABEL = {"national": "France entière (aide nationale)", "european": "Union européenne (aide européenne)"}


def expand_placeholders(text: str) -> str:
    return PLACEHOLDER_RE.sub(lambda m: PLACEHOLDERS[m.group()], text)


def approx_tokens(text: str) -> int:
    return round(len(text) / CHARS_PER_TOKEN)


def capped(values: list[str], limit: int) -> str:
    values = list(dict.fromkeys(v for v in values if v))
    extra = f" (+{len(values) - limit} autres)" if len(values) > limit else ""
    text = ", ".join(values[:limit])
    if len(text) > MAX_LABEL_CHARS:  # free-text areas can list dozens of intercommunalities
        text = text[:MAX_LABEL_CHARS].rsplit(" ", 1)[0] + " (...)"
    return text + extra


def area(aid: dict) -> str:
    if aid["scope"] in SCOPE_LABEL:
        return SCOPE_LABEL[aid["scope"]]
    labels = capped(aid["territory_labels"], MAX_LABELS)
    regions = ", ".join(aid["regions"])
    if regions and labels and labels != regions:
        return f"{labels} (région : {regions})"
    return regions or labels or "territoire non précisé"


def header(aid: dict) -> str:
    lines = [
        f"Aide : {aid['title']}",
        f"Type d'aide : {', '.join(aid['natures']) or 'non précisé'}",
        f"Zone géographique : {area(aid)}",
        f"Financeurs : {capped(aid['financeurs'], MAX_LABELS)}",
        f"Thématiques : {capped(aid['project_categories'], MAX_LABELS)}",
        f"Date limite : {aid['date_fin'] or 'aucune (dispositif permanent)'}",
        f"Fiche mise à jour le : {aid['aid_validation']}",
    ]
    return "\n".join(lines)


def split_long(text: str, budget: int) -> list[str]:
    """Split one long section on sentence boundaries, each piece starting with ~OVERLAP of the previous.

    Each sentence keeps its original trailing whitespace, so line breaks of lists survive the split.
    """
    parts = re.split(f"({SENTENCE_RE.pattern})", text)
    sentences = [s + sep for s, sep in zip(parts[::2], parts[1::2] + [""]) if s.strip()]
    pieces, current = [], []
    for sentence in sentences:
        if current and len("".join(current + [sentence]).strip()) > budget:
            pieces.append("".join(current).strip())
            tail, size = [], 0
            for prev in reversed(current):
                if size + len(prev) > OVERLAP * budget:
                    break
                tail.insert(0, prev)
                size += len(prev)
            current = tail
        current.append(sentence)
    if current:
        pieces.append("".join(current).strip())
    # a single sentence longer than the budget is cut hard (rare: unpunctuated lists)
    out = []
    step = int(budget * (1 - OVERLAP))
    for piece in pieces:
        out.extend(piece[i : i + budget] for i in range(0, max(len(piece) - int(budget * OVERLAP), 1), step))
    return out


def body_blocks(aid: dict, budget: int) -> list[str]:
    """Labelled sections, packed greedily into blocks of at most `budget` characters."""
    sections = []
    for field, label in SECTIONS:
        text = expand_placeholders(aid[field] or "").strip()
        if not text:
            continue
        block = f"{label} : {text}"
        if len(block) <= budget:
            sections.append(block)
        else:
            sections.extend(f"{label} (suite) : {p}" if i else f"{label} : {p}"
                            for i, p in enumerate(split_long(text, budget - len(label) - 12)))
    blocks, current = [], ""
    for section in sections:
        if current and len(current) + 2 + len(section) > budget:
            blocks.append(current)
            current = section
        else:
            current = f"{current}\n\n{section}" if current else section
    if current:
        blocks.append(current)
    return blocks


def chunk_aid(aid: dict, budget: int = MAX_BODY_CHARS) -> list[dict]:
    head = header(aid)
    blocks = body_blocks(aid, budget)
    return [
        {
            "chunk_id": f"{aid['id_aid']}-{i}",
            "id_aid": aid["id_aid"],
            "chunk_index": i,
            "n_chunks": len(blocks),
            "title": aid["title"],
            "header": head,
            "body": block,
            "text": f"{head}\n\n{block}",
            "approx_tokens": approx_tokens(f"{head}\n\n{block}"),
            # metadata kept for filters and tool calls
            "scope": aid["scope"],
            "regions": aid["regions"],
            "natures": aid["natures"],
            "effectif": aid["effectif"],
            "date_fin": aid["date_fin"],
            "aid_validation": aid["aid_validation"],
        }
        for i, block in enumerate(blocks)
    ]


def read_corpus(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_chunks(aids: list[dict], budget: int = MAX_BODY_CHARS) -> list[dict]:
    return [chunk for aid in aids for chunk in chunk_aid(aid, budget)]


def write_jsonl(rows: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=Path("data/processed/aids.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/chunks.jsonl"))
    ap.add_argument("--max-body-chars", type=int, default=MAX_BODY_CHARS)
    args = ap.parse_args()

    aids = read_corpus(args.corpus)
    chunks = build_chunks(aids, args.max_body_chars)
    write_jsonl(chunks, args.out)
    sizes = sorted(c["approx_tokens"] for c in chunks)
    multi = sum(1 for a in {c["id_aid"]: c["n_chunks"] for c in chunks}.values() if a > 1)
    print(f"{len(aids)} aids -> {len(chunks)} chunks ({multi} aids split); "
          f"approx tokens: median {sizes[len(sizes) // 2]}, max {sizes[-1]} -> {args.out}")


if __name__ == "__main__":
    main()
