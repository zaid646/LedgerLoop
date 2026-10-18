"""Phase 6 — render the QLoRA evidence figures (no GPU, no re-run).

Reads the already-produced artefacts and writes the two evidence PNGs:

    evidence/png/13-qlora-eval.png   base-vs-adapter accuracy + per-category
    evidence/png/14-qlora-serve.png  served /classify predictions (table + latency)

Inputs (all already on disk):
    runs/qlora_eval.json        base/adapter accuracy + per-row detail
    runs/qlora_train_meta.json  train_loss, rows, minutes
    evidence/logs/qlora.txt     captured POST /classify responses

Run:
    uv run --with matplotlib python gpu/make_qlora_figures.py
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

BG = "#0f1419"
FG = "#e6edf3"
ACCENT_A = "#8b949e"
ACCENT_B = "#2f81f7"


def _style(ax):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color("#30363d")
    ax.tick_params(colors=FG, labelsize=9)
    ax.title.set_color(FG)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)


def fig_eval(eval_path: Path, meta_path: Path, out: Path) -> None:
    d = json.loads(eval_path.read_text())
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    detail = d.get("detail", [])

    per_cat: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in detail:
        per_cat[r["label"]][1] += 1
        per_cat[r["label"]][0] += 1 if r["ok"] else 0
    cats = sorted(per_cat, key=lambda c: per_cat[c][0] / per_cat[c][1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2), dpi=150)
    fig.patch.set_facecolor(BG)

    # --- left: overall base vs adapter ---
    _style(ax1)
    vals = [d["base_accuracy"], d["adapter_accuracy"]]
    bars = ax1.bar(["base\n(Qwen2.5-0.5B)", "QLoRA\nadapter"], vals, color=[ACCENT_A, ACCENT_B], width=0.55)
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v * 100:.1f}%", ha="center", color=FG, fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 1.08)
    ax1.set_ylabel("held-out accuracy")
    ax1.set_title(f"Base vs fine-tuned  (n={d['val_rows']} held-out pairs)", fontsize=11)
    ax1.annotate(
        f"+{d['delta'] * 100:.1f} pts",
        xy=(1, d["adapter_accuracy"]), xytext=(1, 0.72),
        ha="center", color="#3fb950", fontsize=11, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#3fb950"),
    )

    # --- right: per-category adapter accuracy ---
    _style(ax2)
    ys = range(len(cats))
    accs = [per_cat[c][0] / per_cat[c][1] for c in cats]
    ax2.barh(list(ys), accs, color=ACCENT_B, height=0.62)
    ax2.set_yticks(list(ys))
    ax2.set_yticklabels([c.replace("_", " ").title() for c in cats], fontsize=8)
    ax2.set_xlim(0, 1.05)
    ax2.set_xlabel("adapter accuracy")
    ax2.set_title("Per-GL-category accuracy (adapter)", fontsize=11)

    sub = (
        f"{meta.get('base_model', d['base_model'])}  ·  4-bit NF4 QLoRA  ·  {int(meta.get('epochs', 3))} epochs  ·  "
        f"{meta.get('train_rows', '?')} train rows  ·  train_loss {meta.get('train_loss', float('nan')):.3f}  ·  "
        f"{meta.get('minutes', '?')} min on RTX 4090"
    )
    fig.suptitle("Phase 6 — GL-coding classifier fine-tune", color=FG, fontsize=13, fontweight="bold", y=0.985)
    fig.text(0.5, 0.015, sub, ha="center", color="#8b949e", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(out, facecolor=BG)
    plt.close(fig)
    print(f"wrote {out}")


def _served_rows(log_path: Path) -> list[dict]:
    rows = []
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            m = re.search(r'\{"text":.*?\}', line)
            if m:
                try:
                    rows.append(json.loads(m.group(0)))
                except json.JSONDecodeError:
                    pass
    return rows


def fig_serve(log_path: Path, out: Path) -> None:
    rows = _served_rows(log_path)
    fig, (ax, axl) = plt.subplots(
        1, 2, figsize=(12.5, 5.2), dpi=150, gridspec_kw={"width_ratios": [2.6, 1]}
    )
    fig.patch.set_facecolor(BG)
    _style(ax)
    _style(axl)
    ax.axis("off")

    cell = [[r["text"], r["category"], f"{r['latency_ms']:.0f}"] for r in rows]
    tbl = ax.table(
        cellText=cell,
        colLabels=["input  (description - merchant)", "predicted GL category", "ms"],
        loc="center",
        cellLoc="left",
        colWidths=[0.62, 0.3, 0.08],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.7)
    for (r, c), cellobj in tbl.get_celld().items():
        cellobj.set_edgecolor("#30363d")
        cellobj.set_linewidth(0.6)
        if r == 0:
            cellobj.set_facecolor("#1f6feb")
            cellobj.set_text_props(color="white", fontweight="bold")
        else:
            cellobj.set_facecolor("#161b22" if r % 2 else BG)
            cellobj.set_text_props(color=FG)
            if c == 1:
                cellobj.set_text_props(color="#3fb950", fontweight="bold")
    ax.set_title("Served adapter — POST /classify", color=FG, fontsize=11, pad=14)

    axl.barh([r["text"].split(" - ")[0][:22] for r in rows], [r["latency_ms"] for r in rows], color=ACCENT_B, height=0.6)
    axl.invert_yaxis()
    axl.set_xlabel("latency (ms)")
    axl.set_title("response latency", fontsize=11)

    fig.suptitle("Phase 6 — QLoRA adapter served over HTTP", color=FG, fontsize=13, fontweight="bold", y=0.985)
    fig.text(
        0.5, 0.015,
        "transformers + peft  ·  FastAPI  ·  GET /health + POST /classify  ·  :8001  ·  each call loads base + LoRA adapter",
        ha="center", color="#8b949e", fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(out, facecolor=BG)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default=str(ROOT / "runs/qlora_eval.json"))
    ap.add_argument("--meta", default=str(ROOT / "runs/qlora_train_meta.json"))
    ap.add_argument("--log", default=str(ROOT / "evidence/logs/qlora.txt"))
    ap.add_argument("--out-dir", default=str(ROOT / "evidence/png"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_eval(Path(args.eval), Path(args.meta), out_dir / "13-qlora-eval.png")
    fig_serve(Path(args.log), out_dir / "14-qlora-serve.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
