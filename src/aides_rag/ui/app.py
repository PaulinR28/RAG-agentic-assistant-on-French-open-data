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

# Technical look: white + burgundy, drifting grid, HUD corners, a live diagram of the pipeline, counting figures
CSS = """
<style>
@property --n { syntax: '<integer>'; initial-value: 0; inherits: false; }
:root { --ink: #0E0E10; --bur: #790D16; --bur2: #B0162A; --line: #E6E6E9; --mute: #6B6B73; --soft: #F6F6F7;
        --ease: cubic-bezier(0.32, 0.72, 0, 1); --mono: 'JetBrains Mono', ui-monospace, monospace; }

.stApp { background: #fff; }
.stApp::before { content: ""; position: fixed; inset: -48px; pointer-events: none; z-index: 0;
  background-image: linear-gradient(rgba(14,14,16,0.045) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(14,14,16,0.045) 1px, transparent 1px);
  background-size: 48px 48px; animation: drift 24s linear infinite;
  -webkit-mask-image: radial-gradient(ellipse 80% 60% at 70% 0%, #000 30%, transparent 75%);
          mask-image: radial-gradient(ellipse 80% 60% at 70% 0%, #000 30%, transparent 75%); }
@keyframes drift { to { transform: translate(48px, 48px); } }
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1200px; padding-top: 3.5rem; padding-bottom: 6rem; position: relative; z-index: 1; }

@keyframes rise { from { opacity: 0; transform: translateY(18px); } to { opacity: 1; transform: none; } }
.rise { animation: rise 0.8s var(--ease) both; }
.d1 { animation-delay: 0.08s; } .d2 { animation-delay: 0.16s; } .d3 { animation-delay: 0.24s; }

/* Hero */
.status { display: inline-flex; align-items: center; gap: 0.6rem; font: 500 0.7rem var(--mono); letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--mute); border: 1px solid var(--line); background: #fff; border-radius: 6px; padding: 0.35rem 0.7rem; }
.status i { width: 7px; height: 7px; border-radius: 50%; background: var(--bur2); animation: ping 1.8s var(--ease) infinite; }
@keyframes ping { 0% { box-shadow: 0 0 0 0 rgba(176,22,42,0.55); } 80%, 100% { box-shadow: 0 0 0 9px rgba(176,22,42,0); } }
.title { font-weight: 700; font-size: clamp(3.4rem, 8vw, 6.2rem); line-height: 0.9; letter-spacing: -0.055em; color: var(--ink);
  margin: 1.2rem 0 1rem; }
.title .cur { display: inline-block; width: 0.42em; height: 0.78em; background: var(--bur); margin-left: 0.06em;
  vertical-align: -0.04em; animation: blink 1.05s steps(1) infinite; }
@keyframes blink { 50% { opacity: 0; } }
.lede { max-width: 44rem; font-size: 1.1rem; line-height: 1.55; color: var(--mute); margin: 0; }
.lede b { color: var(--ink); font-weight: 600; }
.lede code { font: 500 0.92em var(--mono); color: var(--bur); background: rgba(121,13,22,0.06); padding: 0.05rem 0.35rem; border-radius: 4px; }

/* How it works, in 3 plain steps: still when idle; while a question runs, one step lights at a time, 3 s each */
.howto { margin: 2.4rem 0 1.6rem; }
.howto .cap { font: 700 0.66rem var(--mono); letter-spacing: 0.16em; text-transform: uppercase; color: var(--mute); margin-bottom: 0.7rem; }
.howto.running .cap { color: var(--bur); }
.howto .row { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1.2rem; }
.howto .st { position: relative; display: flex; align-items: center; gap: 0.9rem; padding: 0.95rem 1.05rem; background: #fff;
  border: 1px solid var(--line); border-radius: 12px; animation: rise 0.8s var(--ease) both; }
.howto .st:not(:last-child)::after { content: "→"; position: absolute; right: -1rem; top: 50%; transform: translateY(-50%);
  color: var(--mute); font-size: 0.9rem; }
.howto .ic { flex: none; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
  font: 700 1rem var(--mono); background: rgba(121,13,22,0.07); color: var(--bur); }
.howto .st b { display: block; font-weight: 600; font-size: 1rem; line-height: 1.25; color: var(--ink); }
.howto .st span { display: block; font-size: 0.8rem; color: var(--mute); margin-top: 0.15rem; }
.howto.running .st { animation: steplit 9s ease-in-out infinite both; }
.howto.running .ic { animation: iclit 9s ease-in-out infinite both; }
@keyframes steplit { 0%, 38%, 100% { border-color: var(--line); background: #fff; }
                     6%, 30% { border-color: var(--bur); background: rgba(121,13,22,0.04); } }
@keyframes iclit { 0%, 38%, 100% { background: rgba(121,13,22,0.07); color: var(--bur); }
                   6%, 30% { background: var(--bur); color: #fff; } }

/* Funnel under each answer: the real numbers of this question */
.funnel { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1.2rem; margin: 0 0 1.8rem; }
.funnel .f { position: relative; padding: 0.9rem 1.05rem; background: #fff; border: 1px solid var(--line); border-radius: 12px;
  animation: rise 0.8s var(--ease) both; }
.funnel .f:not(:last-child)::after { content: "→"; position: absolute; right: -1rem; top: 50%; transform: translateY(-50%);
  color: var(--mute); font-size: 0.9rem; }
.funnel .f .v { font: 700 2.1rem var(--mono); letter-spacing: -0.03em; line-height: 1; color: var(--ink); }
.funnel .f .k { font-size: 0.85rem; color: var(--mute); margin-top: 0.35rem; }
.funnel .f.hot { background: var(--bur); border-color: var(--bur); }
.funnel .f.hot .v, .funnel .f.hot .k { color: #fff; }

/* HUD frame: white panel with burgundy corner ticks */
[data-testid="stForm"], .st-key-answer, .hud { position: relative; background: #fff; border: 1px solid var(--line) !important;
  border-radius: 12px; padding: 1.4rem 1.6rem !important; box-shadow: 0 1px 0 rgba(14,14,16,0.03), 0 24px 48px -36px rgba(14,14,16,0.25); }
[data-testid="stForm"]::before, .st-key-answer::before, .hud::before,
[data-testid="stForm"]::after, .st-key-answer::after, .hud::after { content: ""; position: absolute; width: 14px; height: 14px;
  border-color: var(--bur); border-style: solid; pointer-events: none; }
[data-testid="stForm"]::before, .st-key-answer::before, .hud::before { top: -1px; left: -1px; border-width: 2px 0 0 2px; border-top-left-radius: 12px; }
[data-testid="stForm"]::after, .st-key-answer::after, .hud::after { bottom: -1px; right: -1px; border-width: 0 2px 2px 0; border-bottom-right-radius: 12px; }
.st-key-answer { overflow: hidden; animation: rise 0.8s var(--ease) both; }
.st-key-answer .scan { position: absolute; left: 0; right: 0; top: 0; height: 60px; pointer-events: none;
  background: linear-gradient(180deg, transparent, rgba(121,13,22,0.08), transparent); animation: scan 1.6s var(--ease) 0.2s both; }
@keyframes scan { from { transform: translateY(-60px); } to { transform: translateY(420px); opacity: 0; } }
[data-testid="stForm"] textarea { background: var(--soft) !important; font-size: 1.05rem !important; line-height: 1.55 !important; }
[data-testid="stForm"] [data-baseweb="textarea"] { border-color: transparent !important; }
[data-testid="stForm"] [data-baseweb="textarea"]:focus-within { border-color: var(--bur) !important; }
[data-testid="stForm"] label p { font: 700 0.68rem var(--mono) !important; letter-spacing: 0.14em; text-transform: uppercase; color: var(--mute) !important; }
button[kind="primaryFormSubmit"] { font-family: var(--mono) !important; border-radius: 8px !important; padding: 0.5rem 1.1rem !important;
  border: none !important; transition: transform 0.4s var(--ease), box-shadow 0.4s var(--ease) !important;
  box-shadow: 0 10px 24px -14px rgba(121,13,22,0.9); }
button[kind="primaryFormSubmit"] p { font: 700 0.78rem var(--mono) !important; letter-spacing: 0.1em; text-transform: uppercase; }
button[kind="primaryFormSubmit"]::after { content: "↵"; margin-left: 0.7rem; font-family: var(--mono); opacity: 0.75;
  transition: transform 0.4s var(--ease); }
button[kind="primaryFormSubmit"]:hover::after { transform: translateX(3px); }
button[kind="primaryFormSubmit"]:active { transform: scale(0.98); }

/* Sidebar */
[data-testid="stSidebar"] { border-right: 1px solid var(--line); }
[data-testid="stSidebar"] h3 { font: 700 0.66rem var(--mono) !important; letter-spacing: 0.18em; text-transform: uppercase;
  color: var(--mute) !important; margin-top: 1.2rem !important; }
[data-testid="stSidebar"] h3::before { content: "// "; color: var(--bur); }
[data-testid="stSidebar"] .stButton button { justify-content: flex-start; text-align: left; border-radius: 8px; background: #fff;
  border: 1px solid var(--line); padding: 0.55rem 0.75rem; transition: border-color 0.3s var(--ease), transform 0.3s var(--ease); }
[data-testid="stSidebar"] .stButton button:hover { border-color: var(--bur); transform: translateX(2px); }
[data-testid="stSidebar"] .stButton button[kind="primary"] { background: rgba(121,13,22,0.06); color: var(--ink);
  border-color: var(--bur); box-shadow: inset 3px 0 0 var(--bur); }
[data-testid="stSidebar"] .stButton button p { font-size: 0.84rem; line-height: 1.35; }

/* Result */
.label { font: 700 0.66rem var(--mono); letter-spacing: 0.16em; text-transform: uppercase; color: var(--mute); margin-bottom: 0.6rem; }
.label::before { content: "> "; color: var(--bur); }
.query { font-weight: 600; font-size: clamp(1.5rem, 2.8vw, 2.1rem); line-height: 1.2; letter-spacing: -0.025em; color: var(--ink); margin: 0 0 0.9rem; }
.chips { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-bottom: 1.6rem; }
.chip { font: 500 0.68rem var(--mono); letter-spacing: 0.04em; padding: 0.25rem 0.6rem; border-radius: 5px;
  border: 1px solid var(--line); background: #fff; color: var(--ink); }
.chip.on { border-color: var(--bur); color: var(--bur); }
.chip.warn { background: var(--bur); border-color: var(--bur); color: #fff; }
.sep { height: 1px; background: var(--line); margin: 3.2rem 0 1.4rem; position: relative; }
.sep::after { content: ""; position: absolute; left: 0; top: 0; height: 1px; width: 100%; background: var(--bur);
  transform-origin: left; animation: grow 1.2s var(--ease) both; }

.stats { display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; }
.tile { background: #fff; border: 1px solid var(--line); border-radius: 10px; padding: 0.85rem 1rem; animation: rise 0.8s var(--ease) both;
  transition: border-color 0.3s var(--ease); }
.tile:hover { border-color: var(--bur); }
.tile.wide { grid-column: span 2; }
.tile.hot { background: var(--bur); border-color: var(--bur); color: #fff; }
.tile .k { font: 700 0.6rem var(--mono); letter-spacing: 0.16em; text-transform: uppercase; opacity: 0.6; }
.tile .v { font: 700 1.5rem var(--mono); letter-spacing: -0.03em; margin-top: 0.3rem; }
.tile.hot .v { font-size: 2.8rem; line-height: 1; }
.tile .v small { font-size: 0.75rem; font-weight: 500; opacity: 0.65; margin-left: 0.25rem; }
.count { animation: count 1.6s var(--ease) 0.2s both; counter-reset: n var(--n); }
.count::after { content: counter(n); }
@keyframes count { from { --n: 0; } to { --n: var(--to); } }
.dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 0.5rem; background: #1F9D55;
  vertical-align: middle; animation: ping 1.8s var(--ease) infinite; }
.dot.bad { background: var(--bur2); }

.section { font-weight: 700; font-size: 1.9rem; letter-spacing: -0.03em; color: var(--ink); margin: 0 0 1rem; }
.section span { font: 500 0.72rem var(--mono); color: var(--bur); margin-left: 0.6rem; letter-spacing: 0.08em; vertical-align: middle; }
.sources { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0.7rem; }
.source { position: relative; background: #fff; border: 1px solid var(--line); border-radius: 10px; padding: 1rem 3.4rem 1rem 1.1rem;
  animation: rise 0.8s var(--ease) both; transition: border-color 0.4s var(--ease), transform 0.4s var(--ease); }
.source:hover { border-color: var(--bur); transform: translateY(-2px); }
.source .id { font: 700 0.66rem var(--mono); color: var(--bur); letter-spacing: 0.06em; }
.source .t { font-weight: 600; font-size: 1.02rem; color: var(--ink); margin: 0.3rem 0 0.4rem; line-height: 1.3; }
.source .m { font-size: 0.8rem; color: var(--mute); line-height: 1.45; }
.source a.go { position: absolute; top: 0.9rem; right: 0.9rem; width: 32px; height: 32px; border-radius: 8px; display: flex;
  align-items: center; justify-content: center; border: 1px solid var(--line); color: var(--bur); text-decoration: none;
  font-family: var(--mono); transition: all 0.4s var(--ease); }
.source:hover a.go { background: var(--bur); border-color: var(--bur); color: #fff; }

.steps { position: relative; padding-left: 1.6rem; }
.steps::before { content: ""; position: absolute; left: 0.42rem; top: 0.5rem; bottom: 0.5rem; width: 1px; background: var(--line); }
.step { position: relative; display: grid; grid-template-columns: 10rem minmax(0, 1fr) 5.5rem; gap: 1rem; align-items: center;
  padding: 0.7rem 0; animation: rise 0.8s var(--ease) both; }
.step::before { content: ""; position: absolute; left: -1.45rem; width: 9px; height: 9px; border-radius: 2px; background: #fff; border: 2px solid var(--bur); }
.step.llm::before { background: var(--bur); }
.step > div { min-width: 0; }
.step .n { font: 700 0.76rem var(--mono); color: var(--ink); }
.step .d { font-size: 0.84rem; color: var(--mute); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.step .bar { height: 3px; background: var(--soft); margin-top: 0.45rem; overflow: hidden; border-radius: 2px; }
.step .bar i { display: block; height: 100%; background: var(--bur); transform-origin: left; animation: grow 1.3s var(--ease) both; }
.step.tool .bar i { background: var(--ink); }
@keyframes grow { from { transform: scaleX(0); } to { transform: scaleX(1); } }
.step .ms { font: 700 0.78rem var(--mono); text-align: right; color: var(--ink); }
[data-testid="stExpander"] { margin-top: 0.8rem; }

@media (max-width: 768px) {
  .block-container { padding-top: 2rem; }
  .howto .row, .funnel { grid-template-columns: 1fr; gap: 0.6rem; }
  .howto .st::after, .funnel .f::after { display: none; }
  .sources, .stats { grid-template-columns: 1fr; } .tile.wide { grid-column: span 1; }
  .step { grid-template-columns: 1fr; gap: 0.2rem; } .step .ms { text-align: left; }
}
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation: none !important; } .count { --n: var(--to); } }
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
    st.markdown('<div class="status rise"><i></i>système en ligne · 504 aides indexées · 865 extraits</div>'
                '<div class="title rise d1">Boussole<span class="cur"></span></div>'
                '<p class="lede rise d2">Quelles aides publiques pour le projet <b>vert</b> ou <b>innovant</b> de votre PME ? '
                "Recherche hybride <code>BM25 + bge-m3</code>, réponse générée en local, chaque affirmation citée "
                "et chaque étape tracée.</p>", unsafe_allow_html=True)


HOW_IT_WORKS = [("Je lis votre question", "en français, comme vous l'écrivez"),
                ("Je cherche parmi 504 aides", "par mots-clés et par sens"),
                ("J'écris une réponse sourcée", "chaque aide citée est vérifiée")]


def how_it_works(running: bool = False, wait: str = "") -> str:
    """Three plain steps. While a question runs they light in turn, slowly: a timed animation, not live progress."""
    cap = f"Boussole travaille… ({wait})" if running else "Comment ça marche"
    html = "".join(f'<div class="st" style="animation-delay:{3 * n if running else 0.06 * n}s"><div class="ic" '
                   f'style="animation-delay:{3 * n}s">{n + 1}</div><div><b>{escape(title)}</b><span>{escape(sub)}</span></div></div>'
                   for n, (title, sub) in enumerate(HOW_IT_WORKS))
    return f'<div class="howto{" running" if running else ""}"><div class="cap">{escape(cap)}</div><div class="row">{html}</div></div>'


def funnel(row: dict) -> str:
    """504 aids → found for this question → cited in the answer: the real counts of this row."""
    found, cited = len(row["retrieved_ids"]), len(row["cited_sources"])
    cells = [("", 504, "aides dans la base"),
             ("", found, f"trouvée{'s' if found > 1 else ''} pour votre question"),
             ("hot", cited, f"citée{'s' if cited > 1 else ''} dans la réponse" if cited else "citée : pas de réponse sûre")]
    return '<div class="funnel">' + "".join(
        f'<div class="f {cls}" style="animation-delay:{0.15 * n:.2f}s"><div class="v">{count(v)}</div><div class="k">{escape(k)}</div></div>'
        for n, (cls, v, k) in enumerate(cells)) + "</div>"


def show_sources(row: dict) -> None:
    aids = aids_by_id()
    cited = row["cited_sources"]
    st.markdown(f'<div class="sep"></div><div class="section">Sources<span>[{len(cited)} citée{"s" if len(cited) > 1 else ""}]'
                "</span></div>", unsafe_allow_html=True)
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
        return f'<div class="source"{style}><div class="id">#{escape(aid_id)}</div><div class="m">Absente du corpus.</div></div>'
    url = (aid.get("source_urls") or [None])[0]
    go = f'<a class="go" href="{escape(url)}" target="_blank" title="Fiche officielle">↗</a>' if url else ""
    meta = " · ".join(filter(None, [", ".join(aid.get("financeurs") or []), ", ".join(aid.get("natures") or []),
                                    where(aid),
                                    f"clôture {aid['date_fin']}" if aid.get("date_fin") else "sans date de fin"]))
    return (f'<div class="source"{style}>{go}<div class="id">#ID {escape(aid_id)}</div>'
            f'<div class="t">{escape(aid["title"])}</div><div class="m">{escape(meta)}</div></div>')


def where(aid: dict) -> str:
    places = aid.get("regions") or aid.get("territory_labels") or []
    if len(places) > 3:
        return f"{len(places)} territoires"
    return ", ".join(places) or {"national": "National", "european": "Europe"}.get(aid.get("scope"), "")


def count(value: int) -> str:
    """A number that counts up from 0 on display (CSS only); the title keeps the exact value."""
    return f'<span class="count" style="--to:{int(value)}" title="{int(value)}"></span>'


def stat_tiles(row: dict) -> str:
    ok = row["stop_reason"] == "final_answer" and not row["error"]
    secs = row["latency_ms"] / 1000
    n_tools = len(row["tool_calls"])
    tiles = [("hot wide", "Latence totale", f"{count(secs)}.{int(secs * 10) % 10}<small>s</small>"),
             ("", "Tokens in", count(row["tokens_in"])),
             ("", "Tokens out", count(row["tokens_out"])),
             ("", "Coût", f"{row['cost_usd']:.4f}<small>$</small>"),
             ("", "Appels", f"{count(n_tools)}<small>outil{'s' if n_tools > 1 else ''}</small> "
                            f"{count(row['n_llm_calls'])}<small>llm</small>"),
             ("wide", "Stop reason", f'<span class="dot{"" if ok else " bad"}"></span>{escape(row["stop_reason"] or "?")}')]
    return '<div class="stats">' + "".join(
        f'<div class="tile {cls}" style="animation-delay:{0.06 * n:.2f}s"><div class="k">{k}</div><div class="v">{v}</div></div>'
        for n, (cls, k, v) in enumerate(tiles)) + "</div>"


def show_trace(row: dict) -> None:
    st.markdown('<div class="sep"></div><div class="section">Trace<span>[outils + appels llm]</span></div>',
                unsafe_allow_html=True)
    steps = [("tool", t["name"] + (" (auto)" if t["forced"] else ""),
              f"« {t['args'].get('query', '')} » → {', '.join(t['result_ids']) or 'aucune fiche'}", t["latency_ms"])
             for t in row["tool_calls"]]
    steps += [("llm", f"llm.{c['phase']}", f"{c['tokens_in']:,} → {c['tokens_out']:,} tokens · {c['finish_reason']}"
               .replace(",", " "), c["latency_ms"]) for c in row["llm_calls"]]
    longest = max([s[3] for s in steps] + [1])
    html = "".join(f'<div class="step {kind}" style="animation-delay:{0.08 * n:.2f}s"><div class="n">{escape(name)}</div>'
                   f'<div><div class="d">{escape(desc)}</div><div class="bar"><i style="width:{max(ms / longest * 100, 1.5):.1f}%">'
                   f'</i></div></div><div class="ms">{ms / 1000:.2f} s</div></div>'
                   for n, (kind, name, desc, ms) in enumerate(steps))
    st.markdown(f'<div class="hud"><div class="steps">{html}</div></div>', unsafe_allow_html=True)
    with st.expander("Données brutes (ligne results.jsonl)"):
        st.caption(f"Modèle : {row['model']} · config : {row['config']} · sortie valide : {row['valid_output']}")
        st.json(row, expanded=False)


def show_entry(entry: dict) -> None:
    row, s = entry["row"], entry["settings"]
    st.markdown(f'<div class="sep"></div><div class="label rise">Requête</div>'
                f'<div class="query rise">{escape(row["question"])}</div>', unsafe_allow_html=True)
    chips = [(MODES.get(s.get("mode"), "?"), "on"), (BACKENDS.get(s.get("backend"), "?"), "on"),
             ("fiches entières" if s.get("full_cards") else "extraits", ""), (entry["saved_at"].replace("T", " "), "")]
    html = "".join(f'<span class="chip {cls}">{escape(c)}</span>' for c, cls in chips)
    if row["abstained"]:
        html += '<span class="chip warn">abstention</span>'
    st.markdown(f'<div class="chips rise d1">{html}</div>', unsafe_allow_html=True)
    st.markdown(funnel(row), unsafe_allow_html=True)
    left, right = st.columns([7, 5], gap="large")
    with left, st.container(key="answer"):
        st.markdown('<span class="scan"></span><div class="label">Réponse</div>', unsafe_allow_html=True)
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

    diagram = st.empty()
    diagram.markdown(how_it_works(), unsafe_allow_html=True)
    with st.form("ask", clear_on_submit=False, border=False):
        question = st.text_area("Votre question", height=96,
                                placeholder="Ex. : Je dirige une PME industrielle à Lyon, quelles aides pour "
                                            "réduire ma consommation d'énergie ?")
        asked = st.form_submit_button("Exécuter", type="primary")
    if asked and question.strip():
        llm = LLMClient.from_env(backend=settings["backend"])
        wait = "20 à 60 s en local" if settings["backend"] == "ollama" else "quelques secondes"
        diagram.markdown(how_it_works(running=True, wait=wait), unsafe_allow_html=True)
        with st.spinner(f"Recherche et rédaction ({wait})…"):
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
