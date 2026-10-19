from aides_rag.ui import history


def test_save_then_load_keeps_row_and_settings(tmp_path):
    path = tmp_path / "h.jsonl"
    row = {"question": "Aide pour une PME à Lyon ?", "answer": "Oui", "cited_sources": ["411"], "tool_calls": []}
    settings = {"mode": "rag", "backend": "ollama", "model": "qwen3", "full_cards": True}
    first = history.save(row, settings, path)
    second = history.save(row, settings, path)
    entries = history.load(path)
    assert [e["id"] for e in entries] == [first["id"], second["id"]]
    assert first["id"] != second["id"]
    assert entries[0]["row"] == row and entries[0]["settings"] == settings


def test_load_skips_damaged_lines_and_missing_file(tmp_path):
    path = tmp_path / "h.jsonl"
    assert history.load(path) == []
    history.save({"question": "q"}, {}, path)
    with path.open("a", encoding="utf-8") as f:
        f.write('{"broken": \n')
    assert len(history.load(path)) == 1
