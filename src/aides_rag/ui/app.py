"""Boussole demo UI: ask a question, see the answer, its sources and the agent trace.

    uv run streamlit run src/aides_rag/ui/app.py

A thin layer over `agent.answer()`: the UI only picks the settings, shows the returned row and keeps
every row in the history (data/ui_history.jsonl), so past questions reopen without re-running.
"""

from __future__ import annotations

import os
from html import escape
from functools import lru_cache

import streamlit as st

from aides_rag.agent.agent import answer
from aides_rag.agent.tools import AIDS_PATH
from aides_rag.index.chunks import read_corpus
from aides_rag.llm.client import LLMClient
from aides_rag.ui import history

BACKENDS = {"ollama": "Local · Qwen3 4B", "openrouter": "API · Claude Haiku"}
MODES = {"rag": "RAG (une recherche)", "agent": "Agent (outils)"}

CSS = """
<style>
h1 { color: #790D16; font-weight: 800; letter-spacing: -0.02em; margin-bottom: 0; }
.tagline { color: #6B5A4E; margin-top: 0; }
.source { background: #FFFDF8; border-left: 4px solid #790D16; border-radius: 6px; padding: 0.6rem 0.9rem;
          margin-bottom: 0.5rem; }
.source .meta { color: #6B5A4E; font-size: 0.85rem; }
.badge { display: inline-block; border-radius: 999px; padding: 0.1rem 0.6rem; font-size: 0.8rem;
         background: #AEC4D4; color: #2A1F1D; margin-right: 0.3rem; }
.badge.warn { background: #790D16; color: #F5EFE1; }
[data-testid="stMetricValue"] { font-size: 1.5rem; }
</style>
"""


@lru_cache(maxsize=1)
def aids_by_id() -> dict[str, dict]:
    return {a["id_aid"]: a for a in read_corpus(AIDS_PATH)}


def settings_panel() -> dict:
    st.sidebar.markdown("### Réglages")
    mode = st.sidebar.radio("Mode", list(MODES), format_func=MODES.get, horizontal=True)
    has_key = bool(os.environ.get("OPENROUTER_API_KEY"))
    backends = list(BACKENDS) if has_key else ["ollama"]
    backend = st.sidebar.radio("Modèle", backends, format_func=BACKENDS.get, horizontal=True)
    if not has_key:
        st.sidebar.caption("API indisponible : ajoutez OPENROUTER_API_KEY dans .env.")
    full_cards = st.sidebar.toggle("Fiches entières", value=False,
                                   help="Donne au modèle la fiche complète de chaque aide trouvée, "
                                        "au lieu du meilleur extrait. Désactivé par défaut (mesuré : pas de gain).")
    return {"mode": mode, "backend": backend, "full_cards": full_cards}


def history_panel(entries: list[dict]) -> None:
    st.sidebar.markdown("### Historique")
    if not entries:
        st.sidebar.caption("Aucune question pour l'instant.")
    for e in reversed(entries):
        row, s = e["row"], e["settings"]
        label = row["question"] if len(row["question"]) <= 60 else row["question"][:57] + "…"
        caption = f"{e['saved_at'][5:16].replace('T', ' ')} · {s.get('mode')} · {BACKENDS.get(s.get('backend'), '?')}"
        if st.sidebar.button(label, key=e["id"], help=caption, width="stretch",
                             type="primary" if e["id"] == st.session_state.get("selected") else "secondary"):
            st.session_state.selected = e["id"]
            st.rerun()


def show_sources(row: dict) -> None:
    aids = aids_by_id()
    st.markdown("#### Sources citées")
    if not row["cited_sources"]:
        st.caption("Aucune fiche citée.")
    for aid_id in row["cited_sources"]:
        st.markdown(source_card(aid_id, aids.get(aid_id)), unsafe_allow_html=True)
    others = [i for i in row["retrieved_ids"] if i not in row["cited_sources"]]
    if others:
        with st.expander(f"Fiches trouvées mais non citées ({len(others)})"):
            for aid_id in others:
                st.markdown(source_card(aid_id, aids.get(aid_id)), unsafe_allow_html=True)


def source_card(aid_id: str, aid: dict | None) -> str:
    if aid is None:
        return f'<div class="source"><b>Aide {aid_id}</b> <span class="meta">(absente du corpus)</span></div>'
    url = (aid.get("source_urls") or [None])[0]
    title = f'<a href="{escape(url)}" target="_blank">{escape(aid["title"])}</a>' if url else escape(aid["title"])
    meta = " · ".join(filter(None, [", ".join(aid.get("financeurs") or []), ", ".join(aid.get("natures") or []),
                                    where(aid),
                                    f"clôture {aid['date_fin']}" if aid.get("date_fin") else "sans date de fin"]))
    return f'<div class="source"><b>[{escape(aid_id)}]</b> {title}<br><span class="meta">{escape(meta)}</span></div>'


def where(aid: dict) -> str:
    places = aid.get("regions") or aid.get("territory_labels") or []
    if len(places) > 3:
        return f"{len(places)} territoires"
    return ", ".join(places) or {"national": "National", "european": "Europe"}.get(aid.get("scope"), "")


def show_trace(row: dict) -> None:
    st.markdown("#### Trace")
    c = st.columns(5)
    c[0].metric("Latence", f"{row['latency_ms'] / 1000:.1f} s")
    c[1].metric("Tokens entrée / sortie", f"{row['tokens_in']:,} / {row['tokens_out']:,}".replace(",", " "))
    c[2].metric("Coût", f"{row['cost_usd']:.4f} $")
    c[3].metric("Appels d'outils", len(row["tool_calls"]))
    c[4].metric("Arrêt", row["stop_reason"] or "?")
    with st.expander("Détail : appels d'outils et appels au modèle"):
        st.caption(f"Modèle : {row['model']} · config : {row['config']} · sortie valide : {row['valid_output']}")
        st.markdown("**Outils**")
        st.dataframe([{"outil": t["name"], "arguments": str(t["args"]), "latence (ms)": t["latency_ms"],
                       "fiches": ", ".join(t["result_ids"]), "imposé par le code": t["forced"]}
                      for t in row["tool_calls"]], width="stretch", hide_index=True)
        st.markdown("**Appels au modèle**")
        st.dataframe(row["llm_calls"], width="stretch", hide_index=True)


def show_entry(entry: dict) -> None:
    row, s = entry["row"], entry["settings"]
    st.markdown(f"**Question** · {row['question']}")
    badges = [MODES.get(s.get("mode"), "?"), BACKENDS.get(s.get("backend"), "?"),
              "fiches entières" if s.get("full_cards") else "extraits", entry["saved_at"].replace("T", " ")]
    html = "".join(f'<span class="badge">{b}</span>' for b in badges)
    if row["abstained"]:
        html += '<span class="badge warn">abstention</span>'
    st.markdown(html, unsafe_allow_html=True)
    if row["error"]:
        st.error(f"Échec : {row['error']}")
    else:
        st.markdown("#### Réponse")
        st.markdown(row["answer"])
    show_sources(row)
    show_trace(row)


def main() -> None:
    st.set_page_config(page_title="Boussole", page_icon="🧭", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown("# Boussole")
    st.markdown('<p class="tagline">Quelles aides publiques pour le projet vert ou innovant de votre PME ? '
                "504 dispositifs, réponses sourcées.</p>", unsafe_allow_html=True)

    settings = settings_panel()
    entries = history.load()
    if entries and "selected" not in st.session_state:
        st.session_state.selected = entries[-1]["id"]  # a fresh session opens on the latest question
    history_panel(entries)

    with st.form("ask", clear_on_submit=False):
        question = st.text_area("Votre question", height=90,
                                placeholder="Ex. : Je dirige une PME industrielle à Lyon, quelles aides pour "
                                            "réduire ma consommation d'énergie ?")
        asked = st.form_submit_button("Demander", type="primary")
    if asked and question.strip():
        llm = LLMClient.from_env(backend=settings["backend"])
        wait = "20 à 60 s en local" if settings["backend"] == "ollama" else "quelques secondes"
        with st.spinner(f"Recherche des aides et rédaction de la réponse ({wait})…"):
            row = answer(question.strip(), mode=settings["mode"], llm=llm, full_cards=settings["full_cards"])
        entry = history.save(row, {**settings, "model": llm.model})
        st.session_state.selected = entry["id"]
        st.rerun()
    elif asked:
        st.warning("Écrivez une question.")

    selected = next((e for e in entries if e["id"] == st.session_state.get("selected")), None)
    if selected:
        show_entry(selected)


main()
