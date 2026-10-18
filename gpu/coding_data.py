"""Dataset builder for the Phase 6 GL-coding classifier.

The retrieval corpus in ``ai-service/scripts/index_opensearch.py`` is the source
of truth for merchant -> GL category; this module mirrors a curated subset and
pairs each merchant with category-appropriate descriptions so a small model can
be QLoRA-fine-tuned to map ``<description> - <merchant>`` -> category.

Run standalone (no torch needed):

    python gpu/coding_data.py            # writes gpu/data/coding_{train,val}.jsonl
"""
from __future__ import annotations

import json
import random
from pathlib import Path

CATEGORIES = [
    "TRAVEL",
    "MEALS",
    "OFFICE_SUPPLIES",
    "SOFTWARE",
    "HARDWARE",
    "MARKETING",
    "PROFESSIONAL_SERVICES",
    "UTILITIES",
    "RENT",
    "INSURANCE",
    "TRAINING",
    "ENTERTAINMENT",
    "OTHER",
]

# Mirror of the retrieval seed corpus (merchant -> GL category).
MERCHANTS: dict[str, str] = {
    "Uber": "TRAVEL",
    "Lyft": "TRAVEL",
    "Uber Eats": "MEALS",
    "Delta Air Lines": "TRAVEL",
    "United Airlines": "TRAVEL",
    "Airbnb": "TRAVEL",
    "Marriott Hotels": "TRAVEL",
    "Shell": "TRAVEL",
    "Starbucks": "MEALS",
    "DoorDash": "MEALS",
    "McDonald's": "MEALS",
    "GitHub": "SOFTWARE",
    "Amazon Web Services": "SOFTWARE",
    "Slack": "SOFTWARE",
    "Microsoft": "SOFTWARE",
    "Adobe": "SOFTWARE",
    "Salesforce": "SOFTWARE",
    "Staples": "OFFICE_SUPPLIES",
    "IKEA": "HARDWARE",
    "MR D.I.Y.": "OFFICE_SUPPLIES",
    "Apple": "HARDWARE",
    "Dell": "HARDWARE",
    "Google Ads": "MARKETING",
    "Meta Ads": "MARKETING",
    "Deloitte": "PROFESSIONAL_SERVICES",
    "FedEx": "PROFESSIONAL_SERVICES",
    "WeWork": "RENT",
    "Aetna": "INSURANCE",
    "Coursera": "TRAINING",
    "Netflix": "ENTERTAINMENT",
}

DESCRIPTIONS: dict[str, list[str]] = {
    "TRAVEL": [
        "flight booking", "hotel stay", "rideshare trip", "train ticket", "car rental",
        "fuel purchase", "airport parking", "taxi fare", "road toll", "business trip meal",
        "airline baggage fee", "rail pass",
    ],
    "MEALS": [
        "team lunch", "client dinner", "coffee run", "catering order", "snack delivery",
        "department lunch", "onboarding breakfast", "office coffee beans", "pizza night",
        "conference catering",
    ],
    "OFFICE_SUPPLIES": [
        "printer paper", "pens and notebooks", "binding supplies", "desk organizer",
        "toner cartridge", "sticky notes", "whiteboard markers", "envelopes", "stapler",
        "file folders",
    ],
    "SOFTWARE": [
        "monthly subscription", "annual license", "developer seat", "cloud compute",
        "CI minutes", "observability plan", "design tool license", "API usage",
        "security scanner", "collaboration suite",
    ],
    "HARDWARE": [
        "laptop", "monitor", "office chair", "desk", "keyboard",
        "docking station", "webcam", "server rack", "external SSD", "networking switch",
    ],
    "MARKETING": [
        "ad campaign", "social media ads", "brand promotion", "sponsored post",
        "search ads", "trade show booth", "email marketing tool", "influencer fee",
    ],
    "PROFESSIONAL_SERVICES": [
        "consulting fees", "legal retainer", "courier service", "audit fee",
        "accounting services", "translation service", "recruiting fee", "notary fee",
    ],
    "UTILITIES": [
        "electricity bill", "internet bill", "phone bill", "water bill",
        "heating bill", "gas bill", "waste collection",
    ],
    "RENT": [
        "office rent", "coworking membership", "storage unit", "warehouse lease",
        "parking space rental", "equipment rental",
    ],
    "INSURANCE": [
        "health insurance premium", "liability insurance", "travel insurance",
        "cyber insurance", "property insurance", "workers compensation",
    ],
    "TRAINING": [
        "online course", "certification exam", "conference ticket", "workshop",
        "training materials", "book purchase", "seminar fee", "bootcamp tuition",
    ],
    "ENTERTAINMENT": [
        "client event", "streaming subscription", "game purchase", "team building activity",
        "sports tickets", "concert tickets",
    ],
    "OTHER": ["miscellaneous expense", "general purchase", "unclassified charge", "one-off fee"],
}

PROMPT = "Classify this company expense line into exactly one GL category.\nDescription - Merchant: {text}\nCategory:"


def _rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for merchant, category in MERCHANTS.items():
        rows.append({"text": f"{category.lower().replace('_', ' ')} - {merchant}", "label": category})
        for desc in DESCRIPTIONS[category]:
            rows.append({"text": f"{desc} - {merchant}", "label": category})
    return rows


def build(seed: int = 42, val_frac: float = 0.2) -> tuple[list[dict], list[dict]]:
    """Return (train, val) lists, each item ``{text, label, prompt, target}``."""
    rows = _rows()
    rng = random.Random(seed)
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * val_frac))
    prepared = [
        {"text": r["text"], "label": r["label"], "prompt": PROMPT.format(text=r["text"]), "target": r["label"]}
        for r in rows
    ]
    return prepared[n_val:], prepared[:n_val]


def write(out_dir: Path | str = Path(__file__).parent / "data") -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train, val = build()
    tp = out_dir / "coding_train.jsonl"
    vp = out_dir / "coding_val.jsonl"
    tp.write_text("\n".join(json.dumps(r) for r in train) + "\n")
    vp.write_text("\n".join(json.dumps(r) for r in val) + "\n")
    return tp, vp


if __name__ == "__main__":
    tp, vp = write()
    print(f"wrote {tp} and {vp}")
