"""Find French places in a question and give their region, computed by code (data/processed/places.json).

A small model does not reliably know that Lyon is in Auvergne-Rhône-Alpes, so the prompt states it,
as it already states whether a deadline has passed. Matching rules, to avoid homonyms of common words
("tours", "sens", "orange", "somme"):
  - accents and case are ignored inside a name, but its first letter must be a capital in the question
    (users write "Tours", not "tours" when they mean the city);
  - longest names first, and a matched span is not reused ("Corse-du-Sud" before "Corse");
  - names that are also everyday words (COMMON_WORDS) must follow a place word ("à Tours", "dans la
    Somme"), since a capital at the start of a sentence proves nothing ("Orange propose...").
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path

from aides_rag.retrieval.bm25 import strip_accents

PLACES_PATH = Path(__file__).resolve().parents[3] / "data" / "processed" / "places.json"
SEP = r"[\s\-'’]+"
COMMON_WORDS = {"ORANGE", "SENS", "TOURS", "SOMME", "MANCHE", "LENS", "CHER", "LOT", "VAR", "NORD", "AIN", "GERS",
                "ORNE", "MARNE", "AUBE", "LOIRE", "VIENNE", "CREUSE", "LANDES", "INDRE", "OISE", "EURE", "MEUSE"}
LOCATIVE = re.compile(r"(?i)(?:\b(?:à|a|en|de|du|des|dans|sur|près|pres|vers|le|la|les)|\bl['’]|\bd['’])\s*$")


def _pattern(name: str) -> re.Pattern:
    tokens = [t for t in re.split(SEP, strip_accents(name)) if t]
    first, rest = tokens[0], tokens[1:]
    body = re.escape(first[0].upper()) + (f"(?i:{re.escape(first[1:])})" if len(first) > 1 else "")
    for t in rest:
        body += SEP + f"(?i:{re.escape(t)})"
    return re.compile(rf"(?<![A-Za-z]){body}(?![A-Za-z])")


@lru_cache(maxsize=1)
def _places() -> list[tuple[re.Pattern, list[str], bool]]:
    places = json.loads(PLACES_PATH.read_text(encoding="utf-8"))
    names = sorted(places, key=lambda n: -len(n))  # longest first
    return [(_pattern(n), places[n], strip_accents(n).upper() in COMMON_WORDS) for n in names]


def places_in(question: str) -> list[tuple[str, list[str]]]:
    """[(place as written in the question, its regions)], in order of appearance."""
    text = strip_accents(question)  # same length as the question: spans map back one to one
    taken = [False] * len(text)
    found = []
    for pattern, regions, common in _places():
        for m in pattern.finditer(text):
            if common and not LOCATIVE.search(question[:m.start()]):
                continue
            if not any(taken[m.start():m.end()]):
                taken[m.start():m.end()] = [True] * (m.end() - m.start())
                found.append((m.start(), question[m.start():m.end()], regions))
    return [(place, regions) for _, place, regions in sorted(found)]


def geo_note(question: str) -> str:
    """One prompt line, or "" when the question names no known place (or AIDES_GEO=0, for ablation)."""
    found = places_in(question) if os.getenv("AIDES_GEO", "1") != "0" else []
    if not found:
        return ""
    parts = [f"{place} → {' ou '.join(regions)}" for place, regions in found]
    return ("Repères géographiques (calculés à partir de la question, pour juger si une fiche couvre la zone "
            "de l'entreprise) : " + " ; ".join(parts) + ".")
