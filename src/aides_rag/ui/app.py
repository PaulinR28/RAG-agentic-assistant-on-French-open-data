"""Boussole demo UI: ask a question, see the answer, its sources and the agent trace.

    uv run streamlit run src/aides_rag/ui/app.py

A thin layer over `agent.answer()`: the UI only picks the settings, shows the returned row and keeps
every row in the history (data/ui_history.jsonl), so past questions reopen without re-running.
"""

from __future__ import annotations

import os
from functools import lru_cache
from html import escape

import streamlit as st

from aides_rag.agent.agent import answer
from aides_rag.agent.tools import AIDS_PATH
from aides_rag.index.chunks import read_corpus
from aides_rag.llm.client import LLMClient
from aides_rag.ui import history

BACKENDS = {"ollama": "Local · Qwen3 4B", "openrouter": "API · Claude Haiku"}
MODES = {"rag": "RAG", "agent": "Agent"}

# Editorial luxury with a technical edge: Boussole palette, serif display, mono figures, framed cards
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=JetBrains+Mono:wght@400;500&display=swap');
:root { --ink: #2A1F1D; --bur: #790D16; --sand: #E5D3AF; --cream: #F5EFE1; --sky: #AEC4D4; --muted: #7A6A5E;
        --paper: #FFFDF8; --ease: cubic-bezier(0.32, 0.72, 0, 1);
        --serif: 'Instrument Serif', Georgia, serif; --mono: 'JetBrains Mono', ui-monospace, monospace; }

.stApp { background: radial-gradient(1100px 560px at 88% -12%, rgba(121,13,22,0.08), transparent 62%),
                     radial-gradient(900px 520px at -8% 108%, rgba(174,196,212,0.30), transparent 60%), var(--cream); }
.stApp::after { content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 1; opacity: 0.05;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E"); }
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1180px; padding-top: 4.5rem; padding-bottom: 6rem; }

@keyframes rise { from { opacity: 0; transform: translateY(22px); filter: blur(6px); }
                  to { opacity: 1; transform: none; filter: none; } }
.rise { animation: rise 0.9s var(--ease) both; }
.d1 { animation-delay: 0.08s; } .d2 { animation-delay: 0.16s; } .d3 { animation-delay: 0.24s; }

.eyebrow { display: inline-flex; gap: 0.5rem; align-items: center; border-radius: 999px; padding: 0.3rem 0.8rem;
  font: 500 0.66rem var(--mono); letter-spacing: 0.2em; text-transform: uppercase; color: var(--bur);
  background: rgba(121,13,22,0.06); border: 1px solid rgba(121,13,22,0.12); }
.eyebrow i { width: 6px; height: 6px; border-radius: 50%; background: var(--bur); box-shadow: 0 0 0 4px rgba(121,13,22,0.12); }
.hero h1 { font-family: var(--serif) !important; font-weight: 400 !important; font-size: clamp(3.6rem, 8vw, 6.4rem) !important;
  line-height: 0.92 !important; letter-spacing: -0.03em; color: var(--ink) !important; margin: 1.1rem 0 0.9rem !important; padding: 0 !important; }
.hero h1 em { color: var(--bur); }
.hero p { max-width: 40rem; font-size: 1.12rem; line-height: 1.6; color: var(--muted); margin: 0 0 2.4rem; }
.hero p b { color: var(--ink); font-weight: 600; }

/* Framed surfaces: a paper core sitting in a faint tray (double bezel) */
[data-testid="stForm"], .st-key-answer, .frame {
  background: var(--paper); border: none !important; border-radius: 26px; padding: 1.6rem 1.8rem !important;
  box-shadow: 0 0 0 6px rgba(42,31,29,0.035), 0 0 0 7px rgba(42,31,29,0.07), inset 0 1px 0 #fff,
              0 30px 60px -40px rgba(42,31,29,0.35); }
.st-key-answer { animation: rise 0.9s var(--ease) both; }
[data-testid="stForm"] textarea { background: rgba(229,211,175,0.28) !important; border-radius: 16px !important;
  font-size: 1.05rem !important; line-height: 1.55 !important; }
[data-testid="stForm"] [data-baseweb="textarea"] { border: none !important; border-radius: 16px !important; }
[data-testid="stForm"] label p { font: 500 0.68rem var(--mono) !important; letter-spacing: 0.18em; text-transform: uppercase; color: var(--muted) !important; }

/* Pill button with a nested arrow island */
button[kind="primaryFormSubmit"] { border-radius: 999px !important; padding: 0.45rem 0.45rem 0.45rem 1.4rem !important;
  border: none !important; transition: transform 0.5s var(--ease), box-shadow 0.5s var(--ease) !important;
  box-shadow: 0 14px 30px -16px rgba(121,13,22,0.8); }
button[kind="primaryFormSubmit"]::after { content: "→"; display: inline-flex; align-items: center; justify-content: center;
  width: 30px; height: 30px; margin-left: 0.9rem; border-radius: 50%; background: rgba(245,239,225,0.16);
  transition: transform 0.5s var(--ease); }
button[kind="primaryFormSubmit"]:hover::after { transform: translateX(3px) scale(1.06); }
button[kind="primaryFormSubmit"]:active { transform: scale(0.98); }

/* Sidebar: espresso panel, history as quiet rows */
[data-testid="stSidebar"] h3 { font: 500 0.66rem var(--mono) !important; letter-spacing: 0.22em; text-transform: uppercase;
  color: rgba(245,239,225,0.55) !important; margin-top: 1.4rem !important; }
[data-testid="stSidebar"] .stButton button { justify-content: flex-start; text-align: left; border-radius: 14px;
  border: 1px solid rgba(245,239,225,0.07); background: rgba(245,239,225,0.03); padding: 0.6rem 0.8rem;
  transition: background 0.4s var(--ease), transform 0.4s var(--ease); }
[data-testid="stSidebar"] .stButton button:hover { background: rgba(245,239,225,0.08); transform: translateX(2px); }
[data-testid="stSidebar"] .stButton button[kind="primary"] { background: var(--sand); color: var(--ink); border-color: var(--sand); }
[data-testid="stSidebar"] .stButton button p { font-size: 0.85rem; line-height: 1.35; }

.label { font: 500 0.66rem var(--mono); letter-spacing: 0.2em; text-transform: uppercase; color: var(--muted); margin-bottom: 0.7rem; }
.quote { font-family: var(--serif); font-size: clamp(1.7rem, 3vw, 2.3rem); line-height: 1.15; color: var(--ink); margin: 3.5rem 0 0.9rem; }
.chips { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-bottom: 1.8rem; }
.chip { font: 500 0.68rem var(--mono); letter-spacing: 0.06em; padding: 0.28rem 0.7rem; border-radius: 999px;
  background: rgba(174,196,212,0.45); color: var(--ink); }
.chip.warn { background: var(--bur); color: var(--cream); }

.bento { display: grid; grid-template-columns: 1fr 1fr; gap: 0.6rem; }
.tile { background: var(--paper); border-radius: 20px; padding: 1rem 1.1rem; animation: rise 0.9s var(--ease) both;
  box-shadow: 0 0 0 1px rgba(42,31,29,0.07), 0 20px 40px -32px rgba(42,31,29,0.4); }
.tile.wide { grid-column: span 2; }
.tile.dark { background: var(--ink); color: var(--cream); }
.tile .k { font: 500 0.6rem var(--mono); letter-spacing: 0.2em; text-transform: uppercase; opacity: 0.6; }
.tile .v { font: 500 1.45rem var(--mono); letter-spacing: -0.02em; margin-top: 0.35rem; }
.tile.dark .v { font-size: 2.6rem; }
.tile .v small { font-size: 0.8rem; opacity: 0.6; margin-left: 0.2rem; }
.dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 0.45rem; background: #4E7A62; vertical-align: middle; }
.dot.bad { background: var(--bur); }

.section { font-family: var(--serif); font-size: 2.4rem; line-height: 1.1; color: var(--ink); margin: 4.5rem 0 1.2rem; }
.section span { font: 500 0.75rem var(--mono); color: var(--muted); margin-left: 0.6rem; letter-spacing: 0.1em; }
.sources { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0.9rem; }
.source { position: relative; background: var(--paper); border-radius: 20px; padding: 1.2rem 3.6rem 1.2rem 1.3rem;
  box-shadow: 0 0 0 1px rgba(42,31,29,0.07), 0 20px 40px -34px rgba(42,31,29,0.45); animation: rise 0.9s var(--ease) both;
  transition: transform 0.6s var(--ease), box-shadow 0.6s var(--ease); }
.source:hover { transform: translateY(-3px); box-shadow: 0 0 0 1px rgba(121,13,22,0.25), 0 28px 50px -34px rgba(42,31,29,0.55); }
.source .id { font: 500 0.66rem var(--mono); color: var(--bur); letter-spacing: 0.08em; }
.source .t { font-weight: 700; font-size: 1.02rem; color: var(--ink); margin: 0.3rem 0 0.45rem; line-height: 1.3; }
.source .m { font-size: 0.82rem; color: var(--muted); line-height: 1.45; }
.source a.go { position: absolute; top: 1rem; right: 1rem; width: 34px; height: 34px; border-radius: 50%; display: flex;
  align-items: center; justify-content: center; background: rgba(121,13,22,0.07); color: var(--bur); text-decoration: none;
  transition: transform 0.5s var(--ease), background 0.5s var(--ease); }
.source:hover a.go { transform: translate(2px, -2px) scale(1.06); background: var(--bur); color: var(--cream); }

.steps { position: relative; padding-left: 1.6rem; }
.steps::before { content: ""; position: absolute; left: 0.42rem; top: 0.4rem; bottom: 0.4rem; width: 1px; background: rgba(42,31,29,0.15); }
.step { position: relative; display: grid; grid-template-columns: 9rem 1fr 6rem; gap: 1rem; align-items: center;
  padding: 0.75rem 0; animation: rise 0.9s var(--ease) both; }
.step::before { content: ""; position: absolute; left: -1.45rem; width: 9px; height: 9px; border-radius: 50%;
  background: var(--cream); border: 2px solid var(--bur); }
.step.llm::before { border-color: #5E7F96; }
.step > div { min-width: 0; }
[data-testid="stExpander"] { margin-top: 0.9rem; }
.step .n { font: 500 0.78rem var(--mono); color: var(--ink); }
.step .d { font-size: 0.85rem; color: var(--muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.step .bar { height: 4px; border-radius: 4px; background: rgba(42,31,29,0.08); margin-top: 0.45rem; overflow: hidden; }
.step .bar i { display: block; height: 100%; border-radius: 4px; background: var(--bur); transform-origin: left;
  animation: grow 1.2s var(--ease) both; }
.step.llm .bar i { background: #5E7F96; }
@keyframes grow { from { transform: scaleX(0); } to { transform: scaleX(1); } }
.step .ms { font: 500 0.78rem var(--mono); text-align: right; color: var(--ink); }

@media (max-width: 768px) {
  .block-container { padding-top: 2.5rem; }
  .sources, .bento { grid-template-columns: 1fr; } .tile.wide { grid-column: span 1; }
  .step { grid-template-columns: 1fr; gap: 0.2rem; } .step .ms { text-align: left; }
}
</style>
"""


@lru_cache(maxsize=1)
def aids_by_id() -> dict[str, dict]:
    return {a["id_aid"]: a for a in read_corpus(AIDS_PATH)}


def settings_panel() -> dict:
    st.sidebar.markdown("### Réglages")
    mode = st.sidebar.segmented_control("Mode", list(MODES), format_func=MODES.get, default="rag",
                                        help="RAG : une recherche puis la réponse. Agent : le modèle choisit ses outils.")
    has_key = bool(os.environ.get("OPENROUTER_API_KEY"))
    backends = list(BACKENDS) if has_key else ["ollama"]
    backend = st.sidebar.radio("Modèle", backends, format_func=BACKENDS.get)
    if not has_key:
        st.sidebar.caption("API indisponible : ajoutez OPENROUTER_API_KEY dans .env.")
    full_cards = st.sidebar.toggle("Fiches entières", value=False,
                                   help="Donne au modèle la fiche complète de chaque aide trouvée, "
                                        "au lieu du meilleur extrait. Désactivé par défaut (mesuré : pas de gain).")
    return {"mode": mode or "rag", "backend": backend, "full_cards": full_cards}


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


def hero() -> None:
    st.markdown('<div class="hero"><div class="eyebrow rise"><i></i>RAG hybride · agent · 100 % local</div>'
                '<h1 class="rise d1">Bouss<em>ole</em></h1>'
                '<p class="rise d2">Quelles aides publiques pour le projet <b>vert</b> ou <b>innovant</b> de votre PME ? '
                "504 dispositifs officiels, chaque réponse citée et traçable.</p></div>", unsafe_allow_html=True)


def show_sources(row: dict) -> None:
    aids = aids_by_id()
    cited = row["cited_sources"]
    st.markdown(f'<div class="section">Sources<span>{len(cited)} CITÉE{"S" if len(cited) > 1 else ""}</span></div>',
                unsafe_allow_html=True)
    if not cited:
        st.caption("Aucune fiche citée.")
    else:
        st.markdown(source_grid(cited, aids), unsafe_allow_html=True)
    others = [i for i in row["retrieved_ids"] if i not in cited]
    if others:
        with st.expander(f"Fiches trouvées mais non citées ({len(others)})"):
            st.markdown(source_grid(others, aids), unsafe_allow_html=True)


def source_grid(ids: list[str], aids: dict[str, dict]) -> str:
    return '<div class="sources">' + "".join(source_card(i, aids.get(i), n) for n, i in enumerate(ids)) + "</div>"


def source_card(aid_id: str, aid: dict | None, n: int = 0) -> str:
    style = f' style="animation-delay:{0.06 * n:.2f}s"'
    if aid is None:
        return f'<div class="source"{style}><div class="id">AIDE {escape(aid_id)}</div><div class="m">Absente du corpus.</div></div>'
    url = (aid.get("source_urls") or [None])[0]
    go = f'<a class="go" href="{escape(url)}" target="_blank" title="Fiche officielle">↗</a>' if url else ""
    meta = " · ".join(filter(None, [", ".join(aid.get("financeurs") or []), ", ".join(aid.get("natures") or []),
                                    where(aid),
                                    f"clôture {aid['date_fin']}" if aid.get("date_fin") else "sans date de fin"]))
    return (f'<div class="source"{style}>{go}<div class="id">ID {escape(aid_id)}</div>'
            f'<div class="t">{escape(aid["title"])}</div><div class="m">{escape(meta)}</div></div>')


def where(aid: dict) -> str:
    places = aid.get("regions") or aid.get("territory_labels") or []
    if len(places) > 3:
        return f"{len(places)} territoires"
    return ", ".join(places) or {"national": "National", "european": "Europe"}.get(aid.get("scope"), "")


def stat_tiles(row: dict) -> str:
    ok = row["stop_reason"] == "final_answer" and not row["error"]
    tiles = [("dark wide", "Latence totale", f"{row['latency_ms'] / 1000:.1f}<small>s</small>"),
             ("", "Tokens entrée", f"{row['tokens_in']:,}".replace(",", " ")),
             ("", "Tokens sortie", f"{row['tokens_out']:,}".replace(",", " ")),
             ("", "Coût", f"{row['cost_usd']:.4f}<small>$</small>"),
             ("", "Appels", f"{len(row['tool_calls'])}<small>outil{'s' if len(row['tool_calls']) > 1 else ''}</small> "
                            f"{row['n_llm_calls']}<small>LLM</small>"),
             ("wide", "Arrêt", f'<span class="dot{"" if ok else " bad"}"></span>{escape(row["stop_reason"] or "?")}')]
    return '<div class="bento">' + "".join(
        f'<div class="tile {cls}" style="animation-delay:{0.05 * n:.2f}s"><div class="k">{k}</div><div class="v">{v}</div></div>'
        for n, (cls, k, v) in enumerate(tiles)) + "</div>"


def show_trace(row: dict) -> None:
    st.markdown('<div class="section">Déroulé<span>TRACE DE L\'AGENT</span></div>', unsafe_allow_html=True)
    steps = [("tool", t["name"] + (" ·auto" if t["forced"] else ""),
              f"« {t['args'].get('query', '')} » → {', '.join(t['result_ids']) or 'aucune fiche'}", t["latency_ms"])
             for t in row["tool_calls"]]
    steps += [("llm", f"llm · {c['phase']}", f"{c['tokens_in']:,} → {c['tokens_out']:,} tokens · {c['finish_reason']}"
               .replace(",", " "), c["latency_ms"]) for c in row["llm_calls"]]
    longest = max([s[3] for s in steps] + [1])
    html = "".join(f'<div class="step {kind}" style="animation-delay:{0.07 * n:.2f}s"><div class="n">{escape(name)}</div>'
                   f'<div><div class="d">{escape(desc)}</div><div class="bar"><i style="width:{max(ms / longest * 100, 1.5):.1f}%">'
                   f'</i></div></div><div class="ms">{ms / 1000:.2f} s</div></div>'
                   for n, (kind, name, desc, ms) in enumerate(steps))
    st.markdown(f'<div class="frame"><div class="steps">{html}</div></div>', unsafe_allow_html=True)
    with st.expander("Données brutes (ligne results.jsonl)"):
        st.caption(f"Modèle : {row['model']} · config : {row['config']} · sortie valide : {row['valid_output']}")
        st.json(row, expanded=False)


def show_entry(entry: dict) -> None:
    row, s = entry["row"], entry["settings"]
    st.markdown(f'<div class="quote rise">« {escape(row["question"])} »</div>', unsafe_allow_html=True)
    chips = [MODES.get(s.get("mode"), "?"), BACKENDS.get(s.get("backend"), "?"),
             "fiches entières" if s.get("full_cards") else "extraits", entry["saved_at"].replace("T", " ")]
    html = "".join(f'<span class="chip">{escape(c)}</span>' for c in chips)
    if row["abstained"]:
        html += '<span class="chip warn">abstention</span>'
    st.markdown(f'<div class="chips rise d1">{html}</div>', unsafe_allow_html=True)
    left, right = st.columns([7, 5], gap="large")
    with left, st.container(key="answer"):
        st.markdown('<div class="label">Réponse</div>', unsafe_allow_html=True)
        if row["error"]:
            st.error(f"Échec : {row['error']}")
        else:
            st.markdown(row["answer"])
    with right:
        st.markdown(stat_tiles(row), unsafe_allow_html=True)
    show_sources(row)
    show_trace(row)


def main() -> None:
    st.set_page_config(page_title="Boussole", page_icon="🧭", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    hero()

    settings = settings_panel()
    entries = history.load()
    if entries and "selected" not in st.session_state:
        st.session_state.selected = entries[-1]["id"]  # a fresh session opens on the latest question
    history_panel(entries)

    with st.form("ask", clear_on_submit=False, border=False):
        question = st.text_area("Votre question", height=96,
                                placeholder="Ex. : Je dirige une PME industrielle à Lyon, quelles aides pour "
                                            "réduire ma consommation d'énergie ?")
        asked = st.form_submit_button("Demander à Boussole", type="primary")
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
