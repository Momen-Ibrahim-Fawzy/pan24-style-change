import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from style_change.cosent import PairClassifier, cosent_loss
from style_change.llm import LLMAnnotator, build_messages, parse_changes
from style_change.pan import Document
from style_change.sbert import align_chunks, chunk_text, scores_to_changes, tune_threshold


class LLMParsingTests(unittest.TestCase):
    def test_clean_json(self):
        self.assertEqual(parse_changes('{"changes": [0, 1, 1]}', 3), [0, 1, 1])

    def test_prose_around_json_and_extra_objects(self):
        text = 'Here is the analysis:\n{"changes": [1, 0]}\n{"note": "done"}'
        self.assertEqual(parse_changes(text, 2), [1, 0])

    def test_string_labels_and_single_quotes(self):
        self.assertEqual(parse_changes('{"changes": ["1", "0"]}', 2), [1, 0])
        self.assertEqual(parse_changes("{'changes': [1, 0, 0]}", 3), [1, 0, 0])

    def test_rejects_wrong_length_or_values(self):
        self.assertIsNone(parse_changes('{"changes": [1, 0]}', 3))
        self.assertIsNone(parse_changes('{"changes": [1, 2]}', 2))
        self.assertIsNone(parse_changes("I cannot determine this.", 2))
        self.assertIsNone(parse_changes("", 2))

    def test_messages(self):
        messages = build_messages(["first paragraph", "second", "third"])
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("exactly 2 elements", messages[1]["content"])

    def test_annotator_cache(self):
        doc = Document("5", ["a", "b", "c"], [0, 1], difficulty="easy", split="validation")
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache.jsonl"
            annotator = LLMAnnotator("fake-model", cache_path=str(cache))
            answers = iter(['```json\n{"changes": [0, 1]}\n```'])
            annotator.complete = lambda messages: next(answers)
            self.assertEqual(annotator.annotate(doc), [0, 1])
            self.assertEqual(annotator.annotate(doc), [0, 1])  # served from memory, no second call
            record = json.loads(cache.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(record["key"], "fake-model|easy|validation|5")
            reloaded = LLMAnnotator("fake-model", cache_path=str(cache))
            self.assertEqual(reloaded.annotate(doc), [0, 1])


class SBertHelperTests(unittest.TestCase):
    def test_chunking(self):
        tokenize = lambda t: t.split()
        detok = lambda toks: " ".join(toks)
        text = " ".join(f"w{i}" for i in range(10))
        self.assertEqual(chunk_text(text, tokenize, detok, 4), ["w0 w1 w2 w3", "w4 w5 w6 w7", "w8 w9"])
        self.assertEqual(chunk_text("short text", tokenize, detok, 4), ["short text"])

    def test_cyclic_alignment(self):
        self.assertEqual(align_chunks(["a"], ["x", "y", "z"]), [("a", "x"), ("a", "y"), ("a", "z")])
        self.assertEqual(align_chunks(["a", "b", "c"], ["x", "y"]), [("a", "x"), ("b", "y"), ("c", "x")])

    def test_threshold(self):
        scores = {"1": np.array([0.9, 0.2]), "2": np.array([0.45])}
        self.assertEqual(scores_to_changes(scores, 0.5), {"1": [0, 1], "2": [1]})
        docs = [Document("1", ["a", "b", "c"], [0, 1]), Document("2", ["a", "b"], [0])]
        threshold, f1 = tune_threshold(docs, scores)
        self.assertLessEqual(threshold, 0.45)
        self.assertGreater(threshold, 0.2)
        self.assertAlmostEqual(f1, 1.0)


class CoSENTTests(unittest.TestCase):
    def test_loss_prefers_correct_ranking(self):
        labels = torch.tensor([1, 0, 1, 0])
        good = cosent_loss(torch.tensor([0.9, 0.1, 0.8, 0.2]), labels)
        bad = cosent_loss(torch.tensor([0.1, 0.9, 0.2, 0.8]), labels)
        self.assertLess(good.item(), bad.item())
        self.assertGreater(good.item(), 0.0)

    def test_loss_without_mixed_pairs_is_zero(self):
        # no (positive, negative) combination: only the zero term remains -> log(1) = 0
        self.assertAlmostEqual(cosent_loss(torch.tensor([0.3, 0.7]), torch.tensor([1, 1])).item(), 0.0)

    def test_loss_matches_formula(self):
        cos = torch.tensor([0.6, 0.4])
        labels = torch.tensor([1, 0])
        expected = np.log(1 + np.exp(20 * (0.4 - 0.6)))
        self.assertAlmostEqual(cosent_loss(cos, labels).item(), expected, places=5)

    def test_gradient_flows(self):
        cos = torch.tensor([0.2, 0.7], requires_grad=True)
        cosent_loss(cos, torch.tensor([1, 0])).backward()
        self.assertLess(cos.grad[0].item(), 0)   # raise the positive pair's similarity
        self.assertGreater(cos.grad[1].item(), 0)  # lower the negative pair's similarity

    def test_classifier_shapes(self):
        head = PairClassifier(dim=8)
        u, v = torch.randn(5, 8), torch.randn(5, 8)
        self.assertEqual(head(u, v).shape, (5, 2))
        self.assertEqual(PairClassifier.features(u, v).shape, (5, 24))


if __name__ == "__main__":
    unittest.main()
