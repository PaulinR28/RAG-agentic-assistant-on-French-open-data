"""The agent's tools: their JSON specs (what the model sees) and their Python implementations.

- search_aids(query, region?, open_on?, nature?): the retrieval contract, with metadata filters.
- get_aid(id_aid): the whole aid card, for details that sit outside the retrieved chunk.

Arguments come from a small model, so they are checked: an unknown region or type, or a malformed
date, is dropped and the result says so, letting the model correct itself on its next call.
Aids already shown in full are not repeated, to save context (8k tokens on the demo machine).
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path

from aides_rag.agent.prompts import _status, format_hits
from aides_rag.index.chunks import chunk_aid, read_corpus
from aides_rag.retrieval.bm25 import strip_accents
from aides_rag.retrieval.search import search

AIDS_PATH = Path(__file__).resolve().parents[3] / "data" / "processed" / "aids.jsonl"
SEARCH_K = 5
MAX_AID_CHARS = 8000  # longest card is ~7,800 characters (~2,200 tokens)

REGIONS = ["Auvergne-Rhône-Alpes", "Bourgogne-Franche-Comté", "Bretagne", "Centre-Val de Loire", "Corse",
           "Grand Est", "Hauts-de-France", "Normandie", "Nouvelle-Aquitaine", "Occitanie", "Outre-mer",
           "Pays de la Loire", "Provence-Alpes-Côte d'Azur", "Île-de-France"]
NATURES = ["Subvention", "Appel à projet", "Avance remboursable", "Prêt", "Allègement fiscal",
           "Participation en capital", "Prêt d'honneur", "Garantie", "Exonération de charges sociales"]

TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "search_aids",
        "description": "Cherche dans la base des 504 aides publiques aux entreprises. Renvoie les fiches les plus "
                       "pertinentes avec leur id_aid. Utilise des mots-clés précis : nom de l'aide, type de projet, secteur.",
        "parameters": {"type": "object", "required": ["query"], "properties": {
            "query": {"type": "string", "description": "Mots-clés de recherche, en français."},
            "region": {"type": "string", "enum": REGIONS,
                       "description": "Région de l'entreprise, seulement si la question la donne (une ville = sa région). "
                                      "Les aides nationales et européennes restent incluses."},
            "open_on": {"type": "string",
                        "description": "Date du jour AAAA-MM-JJ, si la question la donne : chaque fiche indique alors "
                                       "si elle est ouverte ou clôturée à cette date. Aucune fiche n'est retirée."},
            "nature": {"type": "string", "enum": NATURES,
                       "description": "Type d'aide, seulement si la question le précise (ex. un prêt)."},
        }},
    }},
    {"type": "function", "function": {
        "name": "get_aid",
        "description": "Renvoie la fiche complète d'une aide déjà trouvée (objet, dépenses, conditions, montant, "
                       "bénéficiaires), quand l'extrait de search_aids ne contient pas le détail cherché.",
        "parameters": {"type": "object", "required": ["id_aid"], "properties": {
            "id_aid": {"type": "string", "description": "Identifiant de l'aide, ex. \"12636\"."},
        }},
    }},
]


@lru_cache(maxsize=1)
def _aids() -> dict[str, dict]:
    return {a["id_aid"]: a for a in read_corpus(AIDS_PATH)}


def _match(value, allowed: list[str]) -> str | None:
    """Accept "bretagne" or "Ile-de-France" for "Bretagne" / "Île-de-France"."""
    key = strip_accents(str(value)).casefold().strip()
    return next((a for a in allowed if strip_accents(a).casefold() == key), None)


def run_tool(name: str, args: dict, shown: set[str]) -> tuple[str, list[str]]:
    """Execute one tool call. Returns the text for the model and the aid ids it exposes.
    `shown` holds the aids already given in full in this conversation; it is updated."""
    if name == "search_aids":
        return _search_aids(args, shown)
    if name == "get_aid":
        return _get_aid(args, shown)
    return f"Outil inconnu : {name}. Outils disponibles : search_aids, get_aid.", []


def _search_aids(args: dict, shown: set[str]) -> tuple[str, list[str]]:
    query = str(args.get("query") or "").strip()
    if not query:
        return "Paramètre query manquant : donne des mots-clés de recherche.", []
    notes, filters, day = [], {}, None
    if args.get("region"):
        filters["region"] = _match(args["region"], REGIONS)
        if not filters["region"]:
            notes.append(f"Région « {args['region']} » inconnue, filtre ignoré. Valeurs possibles : {', '.join(REGIONS)}.")
    if args.get("open_on"):
        # a status annotation, not a filter: filtering hid the very aid asked about (dev q016, closed 5 days before)
        try:
            day = date.fromisoformat(str(args["open_on"])).isoformat()
        except ValueError:
            notes.append(f"Date « {args['open_on']} » invalide (format AAAA-MM-JJ), statut non calculé.")
    if args.get("nature"):
        nature = _match(args["nature"], NATURES)
        filters["natures"] = [nature] if nature else None
        if not nature:
            notes.append(f"Type « {args['nature']} » inconnu, filtre ignoré.")
    hits = search(query, k=SEARCH_K, **{k: v for k, v in filters.items() if v})
    new = [h for h in hits if h.id_aid not in shown]
    again = [h for h in hits if h.id_aid in shown]
    shown.update(h.id_aid for h in new)
    text = "\n".join(notes + [format_hits(new, status_on=day) if new or not again else ""]).strip()
    if again:
        # the earlier display may have had no status, or another date: give it for the date asked now
        listed = [f"{h.id_aid} ({_status(h, day).strip()})" if day else h.id_aid for h in again]
        text += f"\n\nAussi pertinentes, déjà fournies plus haut : {', '.join(listed)}."
    return text, [h.id_aid for h in hits]


def _get_aid(args: dict, shown: set[str]) -> tuple[str, list[str]]:
    id_aid = "".join(ch for ch in str(args.get("id_aid") or "") if ch.isdigit())
    aid = _aids().get(id_aid)
    if not aid:
        return f"Aucune aide avec id_aid « {args.get('id_aid')} ». Utilise un id_aid renvoyé par search_aids.", []
    shown.add(id_aid)
    full = chunk_aid(aid, budget=100_000)[0]["text"][:MAX_AID_CHARS]
    return f'<aide id_aid="{id_aid}">\n{full}\n</aide>', [id_aid]
