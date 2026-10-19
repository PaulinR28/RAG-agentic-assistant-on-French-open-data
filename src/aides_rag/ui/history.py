"""Every question asked in the UI, kept on disk so it can be reopened after a restart.

One JSON line per question in data/ui_history.jsonl (git-ignored): the settings used and the
`agent.answer()` row as is (answer, citations, trace). Reopening an entry never re-runs the model.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

HISTORY_PATH = Path(__file__).resolve().parents[3] / "data" / "ui_history.jsonl"


def save(row: dict, settings: dict, path: Path | None = None) -> dict:
    """Append one entry and return it. `settings`: mode, backend, model, full_cards."""
    path = path or HISTORY_PATH
    saved_at = datetime.now().isoformat(timespec="seconds")
    entry = {"id": f"{saved_at}-{len(load(path))}", "saved_at": saved_at, "settings": settings, "row": row}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def load(path: Path | None = None) -> list[dict]:
    """All entries, oldest first. A damaged line (e.g. the app killed mid-write) is skipped, not fatal."""
    path = path or HISTORY_PATH
    if not path.exists():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries
