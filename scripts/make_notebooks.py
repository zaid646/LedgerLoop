#!/usr/bin/env python3
"""Generate the Phase-1 notebooks (nbformat) so they stay in version control cleanly."""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

REPO_ROOT = Path(__file__).resolve().parents[1]
NB_DIR = REPO_ROOT / "notebooks"

SETUP = """import sys, pathlib
ROOT = pathlib.Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "ai-service"))
sys.path.insert(0, str(ROOT / "evals"))
FIG = ROOT / "notebooks" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
print("repo root:", ROOT)
"""


def nb01() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    c = nb.cells
    c.append(nbf.v4.new_markdown_cell(
        "# 01 · Dataset exploration — SROIE & CORD\n\n"
        "Goal: understand the receipt corpora before building extraction baselines.\n\n"
        "- **SROIE** (`arvindrajan92/sroie_document_understanding`): OCR text with labelled "
        "lines — `company`, `date`, `address`, `total`.\n"
        "- **CORD v2** (`naver-clova-ix/cord-v2`): menu items + total (Indonesian receipts), "
        "used as a distribution-shift stress set.\n\n"
        "Datasets are **streamed** from Hugging Face so nothing large is written to disk."
    ))
    c.append(nbf.v4.new_code_cell(SETUP))
    c.append(nbf.v4.new_code_cell(
        "from app.data import load_sroie, load_cord, parse_amount\n"
        "sroie = load_sroie(120)\n"
        "cord = load_cord(80)\n"
        "print('SROIE records:', len(sroie))\n"
        "print('CORD  records:', len(cord))"
    ))
    c.append(nbf.v4.new_markdown_cell("## Field coverage (how often each gold field is present)"))
    c.append(nbf.v4.new_code_cell(
        "import pandas as pd\n"
        "def coverage(records, fields=('merchant','transaction_date','total_amount')):\n"
        "    return {f: sum(1 for r in records if r.fields.get(f) not in (None,'',[])) for f in fields}\n"
        "cov = pd.DataFrame({'SROIE': coverage(sroie), 'CORD': coverage(cord)})\n"
        "cov['SROIE'] = (cov['SROIE']/len(sroie)*100).round(1)\n"
        "cov['CORD'] = (cov['CORD']/len(cord)*100).round(1)\n"
        "cov"
    ))
    c.append(nbf.v4.new_markdown_cell("## Text length distribution"))
    c.append(nbf.v4.new_code_cell(
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\n"
        "fig, ax = plt.subplots(figsize=(7,4))\n"
        "ax.hist([len(r.text) for r in sroie], bins=20, alpha=.7, label='SROIE')\n"
        "ax.hist([len(r.text) for r in cord], bins=20, alpha=.7, label='CORD')\n"
        "ax.set_xlabel('characters of OCR text'); ax.set_ylabel('# receipts'); ax.legend()\n"
        "ax.set_title('Receipt text length')\n"
        "fig.tight_layout(); fig.savefig(FIG/'text_length.png', dpi=120); plt.show()\n"
        "print('saved', FIG/'text_length.png')"
    ))
    c.append(nbf.v4.new_markdown_cell("## Sample receipts"))
    c.append(nbf.v4.new_code_cell(
        "for r in sroie[:2] + cord[:1]:\n"
        "    print('='*70)\n"
        "    print(r.source, r.id, r.fields)\n"
        "    print(r.text[:400])"
    ))
    c.append(nbf.v4.new_markdown_cell(
        "## Takeaways\n\n"
        "1. SROIE gives clean `company`/`date`/`total` labels — our primary accuracy signal.\n"
        "2. CORD is a harder, foreign-currency set with mostly a `total` label — good for "
        "robustness checks, not headline accuracy.\n"
        "3. Receipts are short (hundreds of chars), so `max-model-len` of 8k is ample."
    ))
    return nb


def nb02() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    c = nb.cells
    c.append(nbf.v4.new_markdown_cell(
        "# 02 · Baseline extraction on the golden set\n\n"
        "Establishes the number the LLM must beat. Two engines:\n\n"
        "- **heuristic** — regex baseline (free, runs here).\n"
        "- **llm** — schema-guided extraction via the self-hosted vLLM. Run it with:\n"
        "  `cd ai-service && uv run python ../evals/run_evals.py --engine llm --golden v0`\n\n"
        "Both write results + W&B runs; this notebook loads and compares them."
    ))
    c.append(nbf.v4.new_code_cell(SETUP))
    c.append(nbf.v4.new_code_cell(
        "import json, sys\n"
        "from run_evals import load_golden, run_heuristic, score\n"
        "records = load_golden('v0')\n"
        "print('golden v0 size:', len(records))"
    ))
    c.append(nbf.v4.new_markdown_cell("## Run the heuristic baseline"))
    c.append(nbf.v4.new_code_cell(
        "preds = run_heuristic(records)\n"
        "summary = score(records, preds)\n"
        "import pandas as pd\n"
        "pd.Series(summary['field_accuracy']).to_frame('accuracy').join(\n"
        "    pd.Series(summary['field_support']).to_frame('support'))"
    ))
    c.append(nbf.v4.new_markdown_cell("## Field accuracy vs exact-document match"))
    c.append(nbf.v4.new_code_cell(
        "import matplotlib; matplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\n"
        "fig, ax = plt.subplots(figsize=(7,4))\n"
        "acc = {k:(v or 0) for k,v in summary['field_accuracy'].items()}; acc['exact_doc']=summary['exact_doc_match']\n"
        "ax.bar(list(acc), list(acc.values()), color='#4C72B0')\n"
        "ax.set_ylim(0,1); ax.set_ylabel('accuracy'); ax.set_title('Heuristic baseline')\n"
        "for i,(k,v) in enumerate(acc.items()): ax.text(i, v+0.02, f'{v:.2f}', ha='center')\n"
        "fig.tight_layout(); fig.savefig(FIG/'heuristic_baseline.png', dpi=120); plt.show()\n"
        "print('exact_doc_match =', summary['exact_doc_match'])"
    ))
    c.append(nbf.v4.new_markdown_cell(
        "## LLM baseline\n\n"
        "Once the GPU/vLLM endpoint is up, run the LLM engine and load "
        "`evals/results/llm-v0-*.json` here to compare accuracy, p95 latency and cost/doc. "
        "The self-healing harness records how many documents were auto-repaired."
    ))
    return nb


def main() -> int:
    NB_DIR.mkdir(parents=True, exist_ok=True)
    nbf.write(nb01(), str(NB_DIR / "01_dataset_exploration.ipynb"))
    nbf.write(nb02(), str(NB_DIR / "02_baseline_extraction.ipynb"))
    print("wrote notebooks:", [p.name for p in sorted(NB_DIR.glob('*.ipynb'))])
    return 0


if __name__ == "__main__":
    sys.exit(main())
