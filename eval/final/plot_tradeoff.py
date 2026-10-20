"""Quality vs latency on held-out, one point per LLM system: eval/final/tradeoff.png.

    uv run --with matplotlib python eval/final/plot_tradeoff.py

Reads eval/final/<system>/summary.json (judged correctness, p50 latency, cost). Latency is on one log axis, so
2 s and 25 s both stay readable; quality is the share of the 20 held-out questions judged correct.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
LOCAL, API = "#A3162A", "#1F78C8"  # validated pair (lightness, chroma, colour-blind separation)
SYSTEMS = [("rag", "rag · Qwen3 4B local (final)", LOCAL, "o", (-12, 22, "right")),
           ("agent", "agent · Qwen3 4B local", LOCAL, "s", (12, -22, "left")),
           ("rag_api", "rag · Claude Haiku (API)", API, "o", (12, -22, "left")),
           ("agent_api", "agent · Claude Haiku (API)", API, "s", (12, 18, "left"))]


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200)
    for name, label, color, marker, (dx, dy, ha) in SYSTEMS:
        s = json.loads((HERE / name / "summary.json").read_text(encoding="utf-8"))
        c = s["judged_correctness"]["heldout"]
        x, y = s["latency_ms"]["p50"] / 1000, 100 * c["correct"] / c["n"]
        cost = s["cost_usd"] / s["n_questions"]
        ax.scatter(x, y, s=90, color=color, marker=marker, edgecolor="white", linewidth=2, zorder=3)
        note = "free" if cost == 0 else f"${cost:.4f} / question"
        ax.annotate(f"{label}\n{c['correct']}/20 correct · {x:.1f} s · {note}", (x, y), xytext=(dx, dy),
                    textcoords="offset points", fontsize=7.5, color="#2A1F1D", va="center", ha=ha)
    ax.set_xscale("log")
    ax.set_xlim(1.2, 120)
    ax.set_ylim(40, 100)
    ax.set_xticks([2, 5, 10, 20, 50])
    ax.minorticks_off()
    ax.set_xticklabels(["2 s", "5 s", "10 s", "20 s", "50 s"])
    ax.set_yticks([40, 50, 60, 70, 80, 90, 100])
    ax.set_yticklabels([f"{t} %" for t in [40, 50, 60, 70, 80, 90, 100]])
    ax.set_xlabel("p50 latency per question (log scale; MacBook Air M3)", fontsize=8, color="#555")
    ax.set_ylabel("held-out questions judged correct", fontsize=8, color="#555")
    ax.set_title("Same retrieval, different answering model: the API reads conditions better, 10× faster",
                 fontsize=9, loc="left", color="#2A1F1D")
    ax.grid(True, color="#EEE", linewidth=0.8)
    ax.tick_params(colors="#555", labelsize=7.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#CCC")
    ax.text(0.0, -0.2, "20 held-out questions: 1 question = 5 points; gaps under ~10 points are not conclusive. "
            "Circle = rag, square = agent.", transform=ax.transAxes, fontsize=6.5, color="#777")
    fig.tight_layout()
    fig.savefig(HERE / "tradeoff.png", facecolor="white")
    print(f"-> {HERE / 'tradeoff.png'}")


if __name__ == "__main__":
    main()
