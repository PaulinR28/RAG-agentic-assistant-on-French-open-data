# data/raw (git-ignored)

Put the 9 raw CSV files of the "Aides entreprises" dataset here:
`aides.csv`, `contacts.csv`, `domaines.csv`, `financeurs.csv`, `natures.csv`,
`niveaux.csv`, `profils.csv`, `projets.csv`, `territoires.csv`.

- Source: https://www.data.gouv.fr/datasets/aides-entreprises (Institut Supérieur des Métiers,
  Licence Ouverte / Open Licence `fr-lo`)
- Direct download: `https://data.aides-entreprises.fr/files/<name>.csv`

The official files are **updated weekly**, so a fresh download will not match the project snapshot
(received 2026-10-13). The corpus is rebuilt from that frozen snapshot. Its SHA-256 checksums are
in `data/processed/selection_report.json` under `raw_sha256`. Ask the team for the snapshot files;
if the checksums differ, the rebuilt corpus may differ too.
