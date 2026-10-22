from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import format_dataset  # noqa: E402
from domain_data import GL_CATEGORIES, build_domain_examples, split_domain_examples  # noqa: E402


class DomainDataTests(unittest.TestCase):
    def test_domain_corpus_is_balanced_and_unique(self) -> None:
        examples = build_domain_examples()
        self.assertEqual(len(examples), 182)
        self.assertEqual({ex.category for ex in examples}, set(GL_CATEGORIES))
        self.assertEqual(len({ex.id for ex in examples}), len(examples))
        self.assertTrue(all(ex.response == ex.category for ex in examples))
        self.assertTrue(all(ex.category in GL_CATEGORIES for ex in examples))

    def test_split_has_no_overlap(self) -> None:
        train, val = split_domain_examples(build_domain_examples(), train_size=152)
        self.assertEqual(len(train), 152)
        self.assertEqual(len(val), 30)
        self.assertFalse({ex.id for ex in train} & {ex.id for ex in val})


class FormatTests(unittest.TestCase):
    def test_prompt_completion_shape(self) -> None:
        record = format_dataset.make_record(
            record_id="x",
            source="ledgerloop",
            instruction="Code this",
            response="MEALS",
            category="MEALS",
        )
        self.assertTrue(record["prompt"].startswith("<|im_start|>system"))
        self.assertTrue(record["prompt"].endswith("<|im_start|>assistant\n"))
        self.assertEqual(record["completion"], "MEALS<|im_end|>\n")
        self.assertEqual(record["text"], record["prompt"] + record["completion"])

    def test_offline_format_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            code = format_dataset.main(["--no-public", "--out-dir", str(out)])
            self.assertEqual(code, 0)
            train = [json.loads(x) for x in (out / "train.jsonl").read_text().splitlines()]
            val = [json.loads(x) for x in (out / "val.jsonl").read_text().splitlines()]
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(len(train), 152)
            self.assertEqual(len(val), 30)
            self.assertEqual(summary["train_by_source"], {"ledgerloop": 152})
            self.assertTrue(all(row["text"] == row["prompt"] + row["completion"] for row in train))


if __name__ == "__main__":
    unittest.main()
