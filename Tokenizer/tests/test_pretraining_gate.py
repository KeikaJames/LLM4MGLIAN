# -*- coding: utf-8 -*-

import json
import os
import tempfile
import unittest

from Tokenizer.evals.pretraining_gate import run_gate
from Tokenizer.tests.test_pretraining_builder import build_smoke_bundle


class PretrainingGateTest(unittest.TestCase):
    def test_smoke_bundle_and_jsonl_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = build_smoke_bundle(tmp)
            bundle_dir = os.path.join(tmp, "bundle")
            bundle.save_dir(bundle_dir)
            inp = os.path.join(tmp, "input.jsonl")
            with open(inp, "w", encoding="utf-8") as f:
                f.write(json.dumps({
                    "type": "image_text",
                    "text": "文字 <image> test",
                    "images": ["x.jpg"],
                    "image_sizes": [[14, 14]],
                }, ensure_ascii=False) + "\n")

            result = run_gate(bundle_dir, inp, max_length=128)

        self.assertTrue(result["passed"], result["failures"])
        self.assertEqual(result["num_samples"], 1)
        self.assertIn("unk_rate", result["metrics"])

    def test_bad_sample_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = build_smoke_bundle(tmp)
            bundle_dir = os.path.join(tmp, "bundle")
            bundle.save_dir(bundle_dir)
            inp = os.path.join(tmp, "bad.jsonl")
            with open(inp, "w", encoding="utf-8") as f:
                f.write(json.dumps({
                    "type": "image_text",
                    "text": "文字 <image> test",
                    "images": [],
                }, ensure_ascii=False) + "\n")

            result = run_gate(bundle_dir, inp, max_length=128)

        self.assertFalse(result["passed"])
        self.assertEqual(result["num_samples"], 0)
        self.assertTrue(any("encode failed" in item["message"] for item in result["failures"]))


if __name__ == "__main__":
    unittest.main()
