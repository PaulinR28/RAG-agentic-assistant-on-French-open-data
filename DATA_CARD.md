# Data card: green-transition & innovation business aids (504 aids)

## 1. Purpose

The corpus feeds an assistant that answers **"Which public funding can my company get?"** for French
SMEs carrying a **green-transition or innovation project**. One document = one aid scheme
(grant, loan, tax relief, call for projects...). The citation unit is the aid id (`id_aid`).

## 2. Source

| | |
|---|---|
| Dataset | Business-aid database ("aides aux entreprises") export, 9 CSV files: `aides`, `contacts`, `domaines`, `financeurs`, `natures`, `niveaux`, `profils`, `projets`, `territoires` |
| Origin | data.gouv.fr — **TODO (team): add the exact dataset URL, publisher and licence** |
| Snapshot | Latest `aid_validation` in the export: 2026-10-09 (raw); files received 2026-10-13 |
| Format | `;`-separated, **cp1252** encoding, quoted multi-line fields, one empty trailing column |
| Integrity | SHA-256 of each raw file in `data/processed/selection_report.json` (`raw_sha256`) |

Raw files are **not committed** (`data/raw/` is git-ignored). Place them there to rebuild.

## 3. Selection pipeline

Script: `src/aides_rag/ingest/select_subset.py` (deterministic; rerun gives a byte-identical output,
checked by `tests/test_corpus.py::test_rebuild_is_deterministic`).

```
uv run python -m aides_rag.ingest.select_subset --raw data/raw --out data/processed
```

| Step | Rule | Kept | Dropped |
|---|---|---:|---:|
| 1. Load | all rows of `aides.csv` | 2,442 | – |
| 2. Quality | combined text of `aid_objet`, `aid_operations_el`, `aid_conditions`, `aid_montant`, `aid_benef` ≥ 600 characters | 2,270 | 172 |
| 3. Dedup | TF-IDF (`min_df=2`) on `objet + operations + montant`, cosine > 0.9 links aids into clusters (computed on all 2,442 aids); keep one aid per cluster: latest `aid_validation`, ties → smallest `id_aid` | 2,018 | 252 |
| 4. Theme | `projets` ids whose parent category (3rd column of `projets.csv`) is **"Transition écologique et énergétique"** or **"Innovation"** | **504** | 1,514 |

**Why each step**
- *Quality*: short cards rarely contain amounts or eligibility conditions, so they cannot answer questions.
- *Dedup*: the export contains template clones (largest cluster: 160 near-identical "Prêt d'honneur
  Initiative [territory]" cards). Clones make retrieval ambiguous: several "correct" documents with the same text.
- *Theme*: a coherent, rule-based scope (not a random sample) that lands close to the ~500 target
  without any arbitrary cut. It covers all regions, national and EU aids.

**Note on the count (503 → 504).** The exploratory analysis reported 503. That run broke ties on
`aid_validation` arbitrarily (unstable sort). In one cluster (3 "Eureka cluster" calls validated
the same day) it kept the off-theme call. The script uses a deterministic tie-break (smallest
`id_aid`), which keeps an in-theme call. Same criteria, deterministic result: **504**.

## 4. Corpus profile (504 aids)

| Dimension | Distribution |
|---|---|
| Scope | 432 territorial, 66 national, 6 European |
| Theme | 384 green only, 104 innovation only, 16 both |
| Aid type (`natures`, multi-valued) | 468 grants, 42 calls for projects, 37 repayable advances, 27 loans, 12 tax reliefs, 7 equity, 3 honour loans, 3 guarantees, 2 social-charge exemptions |
| Regions (multi-valued) | Grand Est 29, Nouvelle-Aquitaine 27, Outre-mer 27, PACA 25, Occitanie 24, Normandie 23, AURA 20, Île-de-France 15, Pays de la Loire 14, Hauts-de-France 13, Bretagne 12, Centre-Val de Loire 12, BFC 11, Corse 3 |
| Deadlines | 168 aids have a `date_fin` (2026: 36, 2027: 107, 2028–2030: 25), all after 2026-10-14; 336 are open-ended |
| Company size | 295 aids exclude companies of 250+ employees |
| Text length | median 1,587 characters, max 7,444; 30 aids above 4,000 |
| Top funders | EU 71 (+ FEDER 24), Bpifrance 45, ADEME 41, regional councils |
| Source links | 406 aids cite at least one URL (524 unique URLs) |

## 5. Output schema (`data/processed/aids.jsonl`, one JSON object per line)

| Field | Content |
|---|---|
| `id_aid` | Aid id (string), the citation key |
| `title`, `objet`, `operations`, `conditions`, `montant`, `beneficiaires` | Cleaned text (control characters removed, whitespace normalised) |
| `scope` | `national` / `european` / `territorial` |
| `regions`, `region_source` | Post-2016 regions (+ `Outre-mer`); how they were found: `territory_ids`, `financeur_postcode`, or `null` |
| `territory_ids`, `territory_labels` | Raw territory ids and their labels, or the raw free-text area name |
| `natures`, `project_categories`, `project_subcategories`, `profils` | Labels resolved from lookup tables |
| `effectif`, `age_entreprise` | Eligible size bands / company age bands |
| `date_fin`, `aid_validation` | ISO dates (`date_fin` null when open-ended; source uses `0000-00-00`) |
| `financeur_ids`, `financeurs`, `contact_ids` | Funders (resolved names) and contact ids |
| `source_urls`, `sources_text` | URLs extracted from the 3 `complements_*` fields, and their raw text |
| `n_chars`, `dedup_cluster_size` | Text length used by step 2; size of the near-duplicate cluster the aid represents |

## 6. Geography: how regions are resolved

1. Each territory id is resolved through the **department** found in itself or its `parents` chain,
   mapped with the **official department → region table** hard-coded in the script (`REGION_DEPTS`).
2. Region ids (`50001`–`50013`) resolve directly. Pre-2016 region ids (e.g. `21` = Picardie),
   which are **missing** from `territoires.csv`, resolve via the departments that list them as parent.
3. If an aid has no resolvable id (free-text intercommunality names), the region is inferred from the
   **postcode of its local funders** (departments, intercommunalities, regional councils).
   National/EU funders and the multi-region Water Agencies are excluded from this inference.
   These aids are flagged `region_source = "financeur_postcode"`.

Result for the 432 territorial aids: 200 resolved from ids, 28 inferred from a funder postcode,
**204 unresolved** (`regions = []`, area kept as text in `territory_labels`).

## 7. Known limitations and data-quality issues

| Issue | Impact | Handling |
|---|---|---|
| **Wrong parent links in `territoires.csv`**: Aisne, Oise, Somme and their communes point to Occitanie (`50007`) instead of Hauts-de-France | Wrong region filters (e.g. an Oise aid tagged Occitanie) | Parent region links ignored; official dept → region table used; regression test added |
| **47% of territorial aids (204) have no machine-readable region** (free-text intercommunality names) | A strict region filter would hide them | Retrieval must not hard-exclude `regions = []`; area name stays searchable in `territory_labels`. Possible fix: map names with the official INSEE EPCI list |
| Region inferred from funder postcode (28 aids) | Approximate for funders located outside the aid's area | Flagged by `region_source` |
| Template placeholders in text: `%mini%` (44 aids, *de minimis* regime) and `%pme%` (16 aids, SME definition) | Unexplained tokens in answers | To expand during chunking (Phase 2) |
| `aid_conditions` empty for 258 aids | Conditions often sit in `beneficiaires` / `montant` instead | Chunk all fields together per aid |
| Dedup threshold also merges **distinct sibling aids** written from one template (e.g. "Fonds Chaleur" biomass vs waste-heat; Eureka cluster calls) | A few real aids are absent | 5 in-theme aids lost because their cluster's representative is off-theme (reported in `selection_report.json`) |
| Source links: of 524 unique URLs, 387 respond, **15 return 404**, 122 could not be verified (bot-blocking 403 on Légifrance/ADEME/regions, TLS/DNS errors) | Some cited links are dead | Citations use `id_aid`; URLs are secondary. Check: `uv run python -m aides_rag.ingest.check_sources` |
| Snapshot data: deadlines and amounts change over time | Answers can be stale | Answers must show the aid's `aid_validation` date |
| Columns dropped: `handicapes`, `femmes`, `jeunes`, `seniors`, `duree_projet` (< 1% filled), `status` (constant) | – | – |
| Corpus is 93% grants | Few questions on loans, guarantees or equity | Golden set deliberately includes loan and tax questions |

## 8. Validation (`uv run pytest`)

The tests check:
- the count matches the report and the step counts chain correctly;
- the output hash matches the report, and a rebuild from raw is byte-identical;
- ids are unique, numeric and present in the raw file;
- every aid is in-theme, has text and has at least 600 characters;
- there are no exact duplicates;
- the vocabularies for region and scope are valid, and `region_source` is consistent;
- dates are ISO, URLs are well formed and funders are resolved;
- regression: Picardie departments resolve to Hauts-de-France;
- the golden set has the right schema, its `gold_docs` exist in the corpus, and out-of-corpus questions have no docs.
