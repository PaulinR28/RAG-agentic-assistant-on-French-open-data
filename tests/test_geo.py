"""Checks on place recognition (retrieval/geo.py) and on full cards in search hits."""

import os

import pytest

from aides_rag.retrieval.geo import geo_note, places_in


@pytest.mark.parametrize("question, expected", [
    ("Je suis artisan à Lyon et je veux un vélo-cargo électrique.", [("Lyon", ["Auvergne-Rhône-Alpes"])]),
    ("Une entreprise à Besançon, dans le Doubs, en Franche-Comté",
     [("Besançon", ["Bourgogne-Franche-Comté"]), ("Doubs", ["Bourgogne-Franche-Comté"]),
      ("Franche-Comté", ["Bourgogne-Franche-Comté"])]),
    ("Nous sommes en Corse-du-Sud.", [("Corse-du-Sud", ["Corse"])]),  # longest name wins over "Corse"
    ("Mon usine près de Tours", [("Tours", ["Centre-Val de Loire"])]),
    ("dans la Somme", [("Somme", ["Hauts-de-France"])]),
])
def test_places_found_with_their_region(question, expected):
    assert places_in(question) == expected


@pytest.mark.parametrize("question", [
    "Les tours de refroidissement sont-elles éligibles ?",   # lowercase: a common word
    "Orange propose une offre. Le sens de la somme.",          # capital at sentence start proves nothing
    "Quel est le taux du crédit d'impôt recherche ?",
])
def test_common_words_are_not_places(question):
    assert places_in(question) == []
    assert geo_note(question) == ""


def test_geo_note_names_the_region():
    assert "Lyon → Auvergne-Rhône-Alpes" in geo_note("Je suis à Lyon")


def test_short_aid_is_shown_whole(monkeypatch):
    """Dev q010: the deciding condition (created before 30 Sept 2023) sits in chunk 3/3 of aid 12844."""
    from aides_rag.retrieval import search as S

    monkeypatch.setenv("AIDES_SEARCH", "bm25")
    question = "Subvention Décarb Flash Artisan : un boulanger créé en 2025 peut-il remplacer son four ?"
    hit = next(h for h in S.search(question, k=5, full_cards=True) if h.id_aid == "12844")
    assert (hit.part, hit.n_parts) == (1, 1) and "30 septembre 2023" in hit.text
    excerpt = next(h for h in S.search(question, k=5, full_cards=False) if h.id_aid == "12844")
    assert excerpt.n_parts == 3 and "30 septembre 2023" not in excerpt.text


def test_full_cards_default_off_and_toggle(monkeypatch):
    """Off by default (no dev gain, +23% tokens); the UI turns it on per question with use_full_cards / answer()."""
    from aides_rag.retrieval import search as S

    monkeypatch.setenv("AIDES_SEARCH", "bm25")
    monkeypatch.delenv("AIDES_FULL_CARDS", raising=False)
    question = "Subvention Décarb Flash Artisan : un boulanger créé en 2025 peut-il remplacer son four ?"
    parts = lambda: next(h.n_parts for h in S.search(question, k=5) if h.id_aid == "12844")
    assert not S.full_cards_enabled() and parts() == 3
    with S.use_full_cards(True):
        assert S.full_cards_enabled() and parts() == 1
    assert parts() == 3  # back to the default after the block
