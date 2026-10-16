"""Build the ~500-aid corpus (green transition + innovation) from the raw aides-entreprises export.

Deterministic pipeline, every step is counted in the selection report:
  1. load        all aids from data/raw/aides.csv
  2. quality     drop aids whose descriptive text is under MIN_CHARS characters
  3. dedup       group near-duplicates (TF-IDF cosine > SIM_THRESHOLD, computed on all aids),
                 keep one aid per group: latest aid_validation, then smallest id_aid
  4. theme       keep aids linked to a THEMES parent category in projets.csv

Usage:
    uv run python -m aides_rag.ingest.select_subset --raw data/raw --out data/processed
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MIN_CHARS = 600
SIM_THRESHOLD = 0.9
THEMES = ("Transition écologique et énergétique", "Innovation")
LENGTH_FIELDS = ["aid_objet", "aid_operations_el", "aid_conditions", "aid_montant", "aid_benef"]
DEDUP_FIELDS = ["aid_objet", "aid_operations_el", "aid_montant"]
SOURCE_FIELDS = ["complements_sources", "complements_formulaires", "complements_reglements"]
FRANCE_ID = "1"
URL_RE = re.compile(r"https?://[^\s<>\"']+")
CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def read_raw(raw: Path, name: str) -> pd.DataFrame:
    df = pd.read_csv(raw / name, sep=";", encoding="cp1252", dtype=str, keep_default_na=False)
    return df.loc[:, ~df.columns.str.startswith("Unnamed")]


def split_ids(value: str) -> list[str]:
    return [t.strip() for t in value.split(",") if t.strip()]


def clean_text(value: str) -> str:
    value = CTRL_RE.sub(" ", value.replace("\r\n", "\n").replace("\r", "\n"))
    value = re.sub(r"[ \t ]+", " ", value)
    return re.sub(r"\n\s*\n+", "\n\n", value).strip()


def to_date(value: str) -> str | None:
    value = value.strip()[:10]
    if not value or value.startswith("0000"):
        return None
    return pd.Timestamp(value).date().isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- lookups


def load_projets(raw: Path) -> tuple[dict[str, str], dict[str, str]]:
    """projets.csv mixes parent-header rows (2 cols) and child rows (id, label, parent, count)."""
    label, parent = {}, {}
    with open(raw / "projets.csv", encoding="cp1252", newline="") as f:
        rows = list(csv.reader(f, delimiter=";"))[1:]
    for row in rows:
        if len(row) >= 3 and row[2]:
            label[row[0]] = row[1]
            parent[row[0]] = row[2]
    return label, parent


# Official department -> region mapping (2016 regions). The source `parents` strings are not
# trusted for this: e.g. Aisne, Oise and Somme are linked to Occitanie (50007) in territoires.csv.
REGION_DEPTS = {
    "Auvergne-Rhône-Alpes": "01 03 07 15 26 38 42 43 63 69 73 74",
    "Bourgogne-Franche-Comté": "21 25 39 58 70 71 89 90",
    "Bretagne": "22 29 35 56",
    "Centre-Val de Loire": "18 28 36 37 41 45",
    "Corse": "2A 2B",
    "Grand Est": "08 10 51 52 54 55 57 67 68 88",
    "Hauts-de-France": "02 59 60 62 80",
    "Île-de-France": "75 77 78 91 92 93 94 95",
    "Normandie": "14 27 50 61 76",
    "Nouvelle-Aquitaine": "16 17 19 23 24 33 40 47 64 79 86 87",
    "Occitanie": "09 11 12 30 31 32 34 46 48 65 66 81 82",
    "Pays de la Loire": "44 49 53 72 85",
    "Provence-Alpes-Côte d'Azur": "04 05 06 13 83 84",
    "Outre-mer": "971 972 973 974 975 976 977 978 986 987 988 97133 97150",
}
DEPT_REGION = {d: r for r, depts in REGION_DEPTS.items() for d in depts.split()}
# New-region ids of territoires.csv (type -1), checked against their labels.
REGION_IDS = {
    "50001": "Grand Est", "50002": "Nouvelle-Aquitaine", "50003": "Bourgogne-Franche-Comté",
    "50004": "Centre-Val de Loire", "50005": "Corse", "50006": "Île-de-France", "50007": "Occitanie",
    "50008": "Hauts-de-France", "50009": "Normandie", "50010": "Pays de la Loire",
    "50011": "Provence-Alpes-Côte d'Azur", "50012": "Auvergne-Rhône-Alpes", "50013": "Bretagne",
}
OVERSEAS_IDS = {"11", "40639"}  # overseas departments / overseas collectivities groupings


class Territories:
    """Resolves territory ids to post-2016 regions through departments.

    An id resolves via the departments (type 3) found in itself or its `parents` chain, mapped with
    the official DEPT_REGION table. Ids that are regions resolve directly (REGION_IDS). Pre-2016 region
    ids are missing from territoires.csv; they resolve to the regions of the departments that list
    them as parent.
    """

    def __init__(self, raw: Path):
        ter = read_raw(raw, "territoires.csv")
        self.label = dict(zip(ter.id_ter, ter.ter_libelle))
        self.parents = {i: split_ids(p) for i, p in zip(ter.id_ter, ter.parents)}
        depts = ter[ter.ter_fk_type == "3"]
        self.dept_code = dict(zip(depts.id_ter, depts.ter_code))
        unknown = set(self.dept_code.values()) - set(DEPT_REGION)
        assert not unknown, f"departments missing from DEPT_REGION: {unknown}"
        self.old = defaultdict(set)
        for dept_id, parents in zip(depts.id_ter, depts.parents):
            for i in split_ids(parents):
                if i not in REGION_IDS and i not in ("0", FRANCE_ID):
                    self.old[i].add(DEPT_REGION[self.dept_code[dept_id]])
        for i in OVERSEAS_IDS:
            self.old[i] = {"Outre-mer"}

    def regions(self, ids: list[str]) -> list[str]:
        out = set()
        for i in ids:
            chain = [i, *self.parents.get(i, [])]
            depts = [self.dept_code[c] for c in chain if c in self.dept_code]
            if depts:
                out.add(DEPT_REGION[depts[0]])
            elif i in REGION_IDS:
                out.add(REGION_IDS[i])
            else:
                out |= self.old.get(i, set()) or next((self.old[c] for c in chain if c in self.old), set())
        return sorted(out)

    @staticmethod
    def regions_from_postcode(cp: str) -> list[str]:
        cp = cp.strip().replace(" ", "")
        if not re.fullmatch(r"\d{5}", cp):
            return []
        dept = cp[:3] if cp.startswith("97") else cp[:2]
        if dept == "20":  # Corsica: both departments share the 20xxx range
            return ["Corse"]
        return [DEPT_REGION[dept]] if dept in DEPT_REGION else []


# ---------------------------------------------------------------- pipeline


def near_duplicate_clusters(df: pd.DataFrame) -> np.ndarray:
    text = df[DEDUP_FIELDS].agg(" ".join, axis=1)
    tfidf = TfidfVectorizer(min_df=2).fit_transform(text)
    sim = cosine_similarity(tfidf)
    np.fill_diagonal(sim, 0)
    _, labels = connected_components(sp.csr_matrix(sim > SIM_THRESHOLD), directed=False)
    return labels


def select(raw: Path) -> tuple[pd.DataFrame, dict, dict[str, str]]:
    df = read_raw(raw, "aides.csv")
    proj_label, proj_parent = load_projets(raw)
    steps = [{"step": "load", "rule": "all rows of aides.csv", "kept": len(df)}]

    df["n_chars"] = df[LENGTH_FIELDS].agg(" ".join, axis=1).str.len()
    df["cluster"] = near_duplicate_clusters(df)
    df["cluster_size"] = df.groupby("cluster").id_aid.transform("size")

    q = df[df.n_chars >= MIN_CHARS]
    steps.append({"step": "quality", "rule": f"combined text of {LENGTH_FIELDS} >= {MIN_CHARS} chars", "kept": len(q), "dropped": len(df) - len(q)})

    q = q.assign(_id=q.id_aid.astype(int)).sort_values(["aid_validation", "_id"], ascending=[False, True])
    d = q.drop_duplicates("cluster", keep="first").drop(columns="_id")
    steps.append({"step": "dedup", "rule": f"one aid per TF-IDF cosine>{SIM_THRESHOLD} cluster on {DEDUP_FIELDS}; keep latest aid_validation, then smallest id_aid", "kept": len(d), "dropped": len(q) - len(d)})

    d = d.assign(project_categories=d.projets.map(lambda v: sorted({proj_parent[i] for i in split_ids(v) if i in proj_parent})))
    s = d[d.project_categories.map(lambda cats: any(t in cats for t in THEMES))]
    steps.append({"step": "theme", "rule": f"projets parent category in {list(THEMES)}", "kept": len(s), "dropped": len(d) - len(s)})

    # Themed aids lost because their cluster representative is outside the themes (reported, not corrected).
    themed = q.projets.map(lambda v: any(proj_parent.get(i) in THEMES for i in split_ids(v)))
    lost = sorted(set(q[themed].cluster) - set(s.cluster))
    report = {
        "steps": steps,
        "themed_aids_lost_to_offtheme_representative": int(q[themed & q.cluster.isin(lost)].shape[0]),
        "raw_sha256": {p.name: sha256(p) for p in sorted(raw.glob("*.csv"))},
    }
    return s.sort_values("id_aid", key=lambda c: c.astype(int)).reset_index(drop=True), report, proj_label


def is_local_funder(echelon: str, name: str) -> bool:
    """Funders whose postcode locates the aid: departments, local bodies and regional councils.

    org_echelon: 1=EU, 2=national, 3=regional (also covers multi-region water agencies),
    4=department, 0/5/6/empty=local. Water agencies and national/EU funders are excluded.
    """
    if echelon == "3":
        return name.startswith("Région")
    return echelon in ("", "0", "4", "5", "6") and name not in ("Financeurs privés", "test")


def build_records(s: pd.DataFrame, raw: Path, proj_label: dict[str, str]) -> pd.DataFrame:
    ter = Territories(raw)
    natures = dict(read_raw(raw, "natures.csv")[["id_typ", "typ_libelle"]].values)
    profils = dict(read_raw(raw, "profils.csv")[["id_tut", "tut_libelle"]].values)
    fin = read_raw(raw, "financeurs.csv").set_index("id_org")

    records = []
    for row in s.itertuples(index=False):
        ter_tokens = split_ids(row.territoires)
        ter_ids = [t for t in ter_tokens if t.isdigit()]
        fin_ids = [i for i in split_ids(row.financeurs) if i in fin.index]
        regions = ter.regions(ter_ids)
        region_source = "territory_ids" if regions else None
        if not regions and row.couverture_geo == "aide territoriale":
            local = [i for i in fin_ids if is_local_funder(fin.at[i, "org_echelon"], fin.at[i, "org_nom"])]
            regions = sorted({r for i in local for r in ter.regions_from_postcode(fin.at[i, "org_cp"])})
            region_source = "financeur_postcode" if regions else None
        sources_text = "\n".join(clean_text(getattr(row, f)) for f in SOURCE_FIELDS if getattr(row, f).strip())
        records.append({
            "id_aid": row.id_aid,
            "title": clean_text(row.aid_nom),
            "objet": clean_text(row.aid_objet),
            "operations": clean_text(row.aid_operations_el),
            "conditions": clean_text(row.aid_conditions),
            "montant": clean_text(row.aid_montant),
            "beneficiaires": clean_text(row.aid_benef),
            "scope": {"aide nationale": "national", "aide européenne": "european"}.get(row.couverture_geo, "territorial"),
            "regions": regions,
            "region_source": region_source,
            "territory_ids": ter_ids,
            "territory_labels": [ter.label.get(t, "") for t in ter_ids] if ter_ids else ([clean_text(row.territoires)] if row.territoires.strip() else []),
            "domaine": row.id_domaine if row.id_domaine not in ("", "0") else None,
            "natures": [natures[i] for i in split_ids(row.natures) if i in natures],
            "project_categories": row.project_categories,
            "project_subcategories": sorted({proj_label[i] for i in split_ids(row.projets) if i in proj_label}),
            "profils": [profils[i] for i in split_ids(row.profils) if i in profils],
            "effectif": split_ids(row.effectif),
            "age_entreprise": split_ids(row.age_entreprise),
            "date_fin": to_date(row.date_fin),
            "aid_validation": to_date(row.aid_validation),
            "financeur_ids": fin_ids,
            "financeurs": [fin.at[i, "org_nom"] for i in fin_ids],
            "contact_ids": split_ids(row.contacts),
            "source_urls": sorted({u.rstrip(".,;)") for u in URL_RE.findall(sources_text)}),
            "sources_text": sources_text,
            "n_chars": int(row.n_chars),
            "dedup_cluster_size": int(row.cluster_size),
        })
    return pd.DataFrame.from_records(records)


def write_jsonl(corpus: pd.DataFrame, path: Path) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for rec in corpus.to_dict(orient="records"):
            # pandas turns None into NaN, which is not valid JSON: write null instead
            rec = {k: None if isinstance(v, float) and math.isnan(v) else v for k, v in rec.items()}
            f.write(json.dumps(rec, ensure_ascii=False, allow_nan=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    args = ap.parse_args()

    selected, report, proj_label = select(args.raw)
    corpus = build_records(selected, args.raw, proj_label)
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / "aids.jsonl"
    write_jsonl(corpus, out)
    report["n_selected"] = len(corpus)
    report["output_sha256"] = sha256(out)
    (args.out / "selection_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    for st in report["steps"]:
        print(f"{st['step']:<8} kept={st['kept']:<5} {st['rule']}")
    print(f"-> {len(corpus)} aids written to {out}")


if __name__ == "__main__":
    main()
