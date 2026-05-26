# -*- coding: utf-8 -*-

import json
import os
import subprocess
import sys
import tempfile
import unittest

from Tokenizer.morphbpe import MorphBPETrainer
from Tokenizer.pretraining import PretrainingDataBuilder, pack_samples
from Tokenizer.unified.bundle import TokenizerBundle


def build_smoke_bundle(tmp: str) -> TokenizerBundle:
    trainer = MorphBPETrainer(vocab_size=200, min_pair_freq=1)
    morphbpe = trainer.train(["ᠮᠣᠩᠭᠣᠯ ᠪᠢᠴᠢᠭ", "ᠮᠣᠩᠭᠣᠯ text"])
    morph_path = os.path.join(tmp, "morphbpe.json")
    morphbpe.save(morph_path)
    bundle = TokenizerBundle.from_files(
        morph_path,
        zh_source="smoke-zh",
        en_source="smoke-en",
        use_smoke_hf=True,
    )
    bundle_dir = os.path.join(tmp, "bundle")
    bundle.save_dir(bundle_dir)
    return TokenizerBundle.from_dir(bundle_dir)


class PretrainingBuilderTest(unittest.TestCase):
    def test_encode_pure_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder = PretrainingDataBuilder(build_smoke_bundle(tmp), max_length=128)
            sample = builder.encode_text("ᠮᠣᠩᠭᠣᠯ 文字 test")
        self.assertEqual(len(sample.input_ids), len(sample.attention_mask))
        self.assertEqual(sample.labels, sample.input_ids)
        self.assertEqual(len(sample.token_offsets), len(sample.input_ids))
        self.assertEqual(sample.modality_spans["image_token_spans"], [])

    def test_encode_image_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder = PretrainingDataBuilder(build_smoke_bundle(tmp), max_length=128)
            sample = builder.encode_json_obj(
                {
                    "type": "image_text",
                    "text": "文字 <image> test",
                    "images": ["x.jpg"],
                    "image_sizes": [[14, 14]],
                }
            )
        self.assertEqual(len(sample.modality_spans["image_token_spans"]), 1)
        self.assertEqual(sample.metadata["images"], ["x.jpg"])
        self.assertEqual(sample.labels, sample.input_ids)

    def test_ocr_metadata_is_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder = PretrainingDataBuilder(build_smoke_bundle(tmp), max_length=128)
            sample = builder.encode_json_obj(
                {
                    "type": "ocr",
                    "text": "<image> text",
                    "images": ["x.jpg"],
                    "image_sizes": [[14, 14]],
                    "ocr": [{"text": "hello", "bbox": [0, 0, 1, 1]}],
                }
            )
        self.assertEqual(sample.metadata["type"], "ocr")
        self.assertEqual(sample.metadata["ocr"][0]["text"], "hello")

    def test_truncation_does_not_cut_image_span(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder = PretrainingDataBuilder(build_smoke_bundle(tmp), max_length=3)
            sample = builder.encode_json_obj(
                {
                    "type": "image_text",
                    "text": "<image> test",
                    "images": ["x.jpg"],
                    "image_sizes": [[29, 29]],
                }
            )
        self.assertLessEqual(len(sample.input_ids), 3)
        self.assertEqual(sample.modality_spans["image_token_spans"], [])
        self.assertTrue(sample.metadata["truncated"])

    def test_pack_samples_for_text_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = build_smoke_bundle(tmp)
            builder = PretrainingDataBuilder(bundle, max_length=64, add_bos=False, add_eos=False)
            first = builder.encode_text("hello")
            second = builder.encode_text("test")
            packed = pack_samples(
                [first, second],
                max_length=64,
                pad_id=bundle.tokenizer.vocab["<pad>"],
                eos_id=bundle.tokenizer.vocab["<eos>"],
            )
        self.assertEqual(len(packed), 1)
        self.assertEqual(packed[0].metadata["num_samples"], 2)
        self.assertLessEqual(len(packed[0].input_ids), 64)
        self.assertEqual(len(packed[0].labels), len(packed[0].attention_mask))

    def test_build_pretraining_data_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = build_smoke_bundle(tmp)
            bundle_dir = os.path.join(tmp, "bundle")
            bundle.save_dir(bundle_dir)
            inp = os.path.join(tmp, "input.jsonl")
            out = os.path.join(tmp, "out.jsonl")
            with open(inp, "w", encoding="utf-8") as f:
                f.write(json.dumps({
                    "type": "image_text",
                    "text": "文字 <image>",
                    "images": ["x.jpg"],
                    "image_sizes": [[14, 14]],
                }, ensure_ascii=False) + "\n")
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "Tokenizer.tools.build_pretraining_data",
                    "--tokenizer-bundle",
                    bundle_dir,
                    "--input",
                    inp,
                    "--output",
                    out,
                    "--max-length",
                    "128",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            summary = json.loads(proc.stdout)
            with open(out, "r", encoding="utf-8") as f:
                row = json.loads(f.readline())
        self.assertEqual(summary["num_samples"], 1)
        self.assertIn("input_ids", row)
        self.assertEqual(len(row["input_ids"]), len(row["labels"]))


if __name__ == "__main__":
    unittest.main()
