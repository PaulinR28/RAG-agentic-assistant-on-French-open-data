"""Build data/processed/places.json: place names a user may write, mapped to post-2016 regions.

Why: a 4B model does not reliably know that Lyon is in Auvergne-Rhône-Alpes (dev q011: the right aid
ranked 1st, the model still answered "no aid in Lyon"). The agent computes the region in code and
states it in the prompt, as it already does for deadlines.

Entries (name as written in territoires.csv -> regions):
  - the 13 regions + Outre-mer, and the pre-2016 regions people still use (Alsace, Picardie...)
  - the 101 departments
  - communes of at least MIN_POPULATION inhabitants (fewer homonyms of common words), plus the
    cities whose population is missing from the source (split into arrondissements: Lyon, Marseille)
A name shared by several places keeps all their regions.

Usage:
    uv run python -m aides_rag.ingest.build_places --raw data/raw --out data/processed/places.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from aides_rag.ingest.select_subset import DEPT_REGION, REGION_DEPTS, Territories, read_raw

MIN_POPULATION = 10_000
MISSING_POPULATION = {"LYON", "MARSEILLE"}  # population 0 in territoires.csv (arrondissements)

OLD_REGIONS = {
    "Alsace": "Grand Est", "Lorraine": "Grand Est", "Champagne-Ardenne": "Grand Est",
    "Aquitaine": "Nouvelle-Aquitaine", "Limousin": "Nouvelle-Aquitaine", "Poitou-Charentes": "Nouvelle-Aquitaine",
    "Languedoc-Roussillon": "Occitanie", "Midi-Pyrénées": "Occitanie",
    "Auvergne": "Auvergne-Rhône-Alpes", "Rhône-Alpes": "Auvergne-Rhône-Alpes",
    "Bourgogne": "Bourgogne-Franche-Comté", "Franche-Comté": "Bourgogne-Franche-Comté",
    "Basse-Normandie": "Normandie", "Haute-Normandie": "Normandie",
    "Nord-Pas-de-Calais": "Hauts-de-France", "Picardie": "Hauts-de-France",
    "Région Centre": "Centre-Val de Loire", "Région Sud": "Provence-Alpes-Côte d'Azur", "PACA": "Provence-Alpes-Côte d'Azur",
    "Île-de-France": "Île-de-France", "IDF": "Île-de-France",
}


def build(raw: Path) -> dict[str, list[str]]:
    ter = read_raw(raw, "territoires.csv")
    regions = Territories(raw)
    places: dict[str, set[str]] = {}

    def add(name: str, region_list) -> None:
        name = name.strip()
        if name and region_list:
            places.setdefault(name, set()).update(region_list)

    for region in REGION_DEPTS:
        add(region, [region])
    for old, new in OLD_REGIONS.items():
        add(old, [new])
    for row in ter[ter.ter_fk_type == "3"].itertuples():
        if row.ter_code in DEPT_REGION:
            add(row.ter_libelle, [DEPT_REGION[row.ter_code]])
    pop = pd.to_numeric(ter.population, errors="coerce").fillna(0)
    communes = ter[(ter.ter_fk_type == "4") & ((pop >= MIN_POPULATION) | ter.ter_libelle.isin(MISSING_POPULATION))]
    for row in communes.itertuples():
        add(row.ter_libelle, regions.regions([row.id_ter]))
    return {name: sorted(r) for name, r in sorted(places.items())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/places.json"))
    args = ap.parse_args()
    places = build(args.raw)
    args.out.write_text(json.dumps(places, ensure_ascii=False, indent=0) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(places)} place names -> {args.out}")


if __name__ == "__main__":
    main()
