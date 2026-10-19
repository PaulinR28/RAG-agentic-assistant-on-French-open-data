# Data card: green-transition & innovation business aids (504 aids)

## 1. Purpose

The corpus feeds an assistant that answers **"Which public funding can my company get?"** for French
SMEs carrying a **green-transition or innovation project**. One document = one aid scheme
(grant, loan, tax relief, call for projects...). The citation unit is the aid id (`id_aid`).

## 2. Source

| | |
|---|---|
| Dataset | Business-aid database ("aides aux entreprises") export, 9 CSV files: `aides`, `contacts`, `domaines`, `financeurs`, `natures`, `niveaux`, `profils`, `projets`, `territoires` |
| Origin | **Confirmed by the project lead (2026-10-14):** data.gouv.fr dataset **"Aides entreprises"**, https://www.data.gouv.fr/datasets/aides-entreprises (dataset id `5ae2d5b5c751df3498bc70ad`). Publisher: **Institut Supérieur des Métiers**. The database aides-entreprises.fr was set up by the French government; files are served from `https://data.aides-entreprises.fr/files/<name>.csv`. The page says it is updated weekly |
| Licence | **Licence Ouverte / Open Licence**, data.gouv.fr licence id `fr-lo` (version 1.0, https://www.etalab.gouv.fr/wp-content/uploads/2014/05/Licence_Ouverte.pdf), as declared on the dataset page, read through the data.gouv.fr API on 2026-10-14. Note: this is *not* the 2.0 version (`lov2`). Reuse requires attribution: cite the source, "Institut Supérieur des Métiers – aides-entreprises.fr", and the date of last update |
| Snapshot | Files received 2026-10-13; latest `aid_validation` in the snapshot: 2026-10-09. Snapshot vs. official files re-downloaded on 2026-10-14: the official files are newer (weekly updates; 8 of 9 checksums differ, `domaines.csv` identical). Columns are identical in all files. `aides.csv` has 2,441 rows (2 aids removed, 1 added). **All 504 selected aids are still present, with unchanged text and deadlines.** The project keeps the 2026-10-13 snapshot frozen for reproducibility |
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

**Note on the count (503 → 504).** The agreed rules said "keep the most recent `aid_validation`" but
did not say what to do on a tie. The exploratory analysis reported 503. It sorted on `aid_validation`
alone with an unstable sort, so ties were resolved by arbitrary row order. 30 near-duplicate clusters
have a tie on their latest date. Only one of them changes the themed count:

| `id_aid` | Aid | `aid_validation` | In theme |
|---|---|---|---|
| 12790 | Appel à Projets du Cluster Eureka ITEA | 2026-09-08 | yes |
| 12791 | Appel à Projets du Cluster Eureka CELTIC-NEXT | 2026-09-08 | yes |
| 12792 | Appel à Projets du Cluster Eureka Eurogia | 2026-09-08 | no |

The exploratory run kept 12792, which is off-theme, so the cluster dropped out of the corpus.
The script breaks ties by smallest `id_aid` and keeps 12790. **12790 is the extra record.** Two other
ties swap the representative without changing the count: 11947 replaces 11949 ("Fonds Chaleur") and
12305 replaces 12712 (biodiversity).

Sensitivity of the count to the tie-break rule: smallest id → 504, longest text → 504, largest
id → 503. The smallest-id rule was chosen because it is neutral, not to reach a target count.

> **Approved rule (project lead, 2026-10-14).** Within a near-duplicate cluster, keep the aid with
> the latest `aid_validation`. If several aids share that latest date, keep the one with the
> **smallest `id_aid`** (compared as an integer). The approved corpus size is **504 aids**.
> Changing this rule changes the corpus and requires a new decision.

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

## 9. Cleaning summary

All steps run in `select_subset.py` (§3) except placeholder expansion (§10).

- **Decoding:** cp1252 → UTF-8; `;` separator with quoted multi-line fields; empty trailing column dropped.
- **Text:** control characters removed, whitespace normalised, in every text field.
- **Dates:** `0000-00-00` → `null` (open-ended), all other dates ISO 8601.
- **Lookups:** ids in `natures`, `projets`, `profils`, `financeurs`, `territoires` resolved to labels; regions resolved as in §6.
- **Columns dropped:** `handicapes`, `femmes`, `jeunes`, `seniors`, `duree_projet` (< 1% filled), `status` (constant).
- **Filtering:** cards under 600 characters removed (172), near-duplicates merged (252), off-theme aids removed (1,514).
- **Links:** URLs extracted from the three `complements_*` fields into `source_urls`.

## 10. From aids to chunks

Script: `src/aides_rag/index/chunks.py` → `data/processed/chunks.jsonl`.

| | |
|---|---|
| Output | **865 chunks** from 504 aids (1.7 per aid on average) |
| Structure | metadata header + body text |
| Header | title, aid type (`natures`), area (scope, or territory labels + resolved region), funders, themes (`project_categories`), deadline (`date_fin`, or "aucune"), card update date (`aid_validation`). Long lists capped at 5 items / 250 characters (one aid lists 613 communes) |
| Body | labelled sections in this order: Objet → Dépenses et opérations éligibles → Conditions → Montant → Bénéficiaires (empty sections skipped) |
| Size | body ≤ 1,800 characters; **390 median / 636 max tokens** with the bge-m3 (XLM-R) tokenizer (3.98 characters/token), so no chunk exceeds bge-m3's 1,024-token window (tested). Qwen3 counts more: 3.49 characters/token |
| Splitting | an aid whose body fits 1,800 characters stays one chunk (281 aids); longer aids are packed section by section; a section longer than the budget is split on sentence and list boundaries, each piece repeating ~15% of the previous one, line breaks kept |
| Placeholders | `%mini%` → "Aide soumise au règlement de minimis : le total des aides de minimis reçues par l'entreprise est plafonné sur trois exercices fiscaux."; `%pme%` → "PME au sens de la définition européenne (moins de 250 salariés, chiffre d'affaires annuel inférieur ou égal à 50 M€ ou total de bilan inférieur ou égal à 43 M€)" |
| Chunk metadata | `chunk_id`, `id_aid`, `chunk_index`, `n_chunks`, `approx_tokens`, plus for filters `scope`, `regions`, `natures`, `effectif`, `date_fin`, `aid_validation` |

**Why this design**
- One aid = one citation unit, and the median aid (1,587 characters) fits in one chunk, so most aids stay whole.
- The header is repeated in every chunk, so a chunk from the middle of a long aid still carries its name, funder, region and deadline. This lets it match and be cited correctly.
- Conditions often sit in `beneficiaires` or `montant` rather than `conditions` (empty for 258 aids, §7), so all fields of an aid are chunked together.
- 20 random aids and 20 random chunks were read before keeping this design (seed 42, `docs/reviews/chunking-review.md`): confirmed, with one fix (line breaks kept when splitting) and two rules passed to the answering model (respect explicit exclusions; say when an amount is not stated).
- Chunk-size and header experiments: `EVALUATION.md` §5 (E1, E2).
