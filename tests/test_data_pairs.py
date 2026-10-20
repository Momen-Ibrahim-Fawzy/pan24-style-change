import json
import tempfile
import unittest
from pathlib import Path

from style_change.metrics import evaluate, f1_per_class, macro_f1
from style_change.pairs import build_pairs, consecutive_pairs, frame_to_pairs, grouped_pairs, pairs_frame
from style_change.pan import (Document, author_segments, load_pan24, read_problem_dir, read_solutions,
                              write_solutions)
from style_change.stats import summary


def write_problem(directory: Path, pid: str, paragraphs, changes, authors):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"problem-{pid}.txt").write_text("\n".join(paragraphs) + "\n", encoding="utf-8")
    (directory / f"truth-problem-{pid}.json").write_text(json.dumps({"authors": authors, "changes": changes}))


# A: p0 p1 | B: p2 | C: p3 p4 p5
DOC = Document("7", ["a1", "a2", "b1", "c1", "c2", "c3"], [0, 1, 1, 0, 0], 3, "hard", "train")


class PanFormatTests(unittest.TestCase):
    def test_read_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_problem(root / "easy" / "train", "10", ["x", "y", "z"], [1, 0], 2)
            write_problem(root / "easy" / "train", "2", ["p", "q"], [1], 2)
            write_problem(root / "hard" / "validation", "1", ["u", "v"], [0], 1)
            docs = read_problem_dir(root / "easy" / "train", "easy", "train")
            self.assertEqual([d.id for d in docs], ["2", "10"])  # numeric order
            self.assertEqual(docs[1].changes, [1, 0])
            data = load_pan24(root)
            self.assertEqual(set(data), {("easy", "train"), ("hard", "validation")})
            table = summary(data)
            self.assertEqual(int(table["pairs"].sum()), 4)

    def test_mismatched_truth(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            write_problem(d, "1", ["x", "y", "z"], [1], 2)
            write_problem(d, "2", ["x", "y"], [1], 2)
            with self.assertRaises(ValueError):
                read_problem_dir(d)
            self.assertEqual([doc.id for doc in read_problem_dir(d, strict=False)], ["2"])

    def test_solutions_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_solutions({"3": [0, 1], "4": [1]}, tmp)
            self.assertEqual(read_solutions(tmp), {"3": [0, 1], "4": [1]})

    def test_author_segments(self):
        self.assertEqual(author_segments([0, 1, 1, 0, 0]), [0, 0, 1, 2, 2, 2])
        self.assertEqual(author_segments([]), [0])


class PairTests(unittest.TestCase):
    def test_consecutive(self):
        pairs = consecutive_pairs(DOC)
        self.assertEqual(len(pairs), 5)
        self.assertEqual([(p.text1, p.text2, p.change) for p in pairs][:2], [("a1", "a2", 0), ("a2", "b1", 1)])
        self.assertEqual(pairs[0].same_author, 1)

    def test_grouped(self):
        pairs = {(p.text1, p.text2, p.change) for p in grouped_pairs(DOC)}
        positives = {(a, b) for a, b, c in pairs if c == 0}
        negatives = {(a, b) for a, b, c in pairs if c == 1}
        self.assertEqual(positives, {("a1", "a2"), ("c1", "c2"), ("c1", "c3"), ("c2", "c3")})
        self.assertEqual(negatives, {("a1", "b1"), ("a2", "b1"), ("b1", "c1"), ("b1", "c2"), ("b1", "c3")})
        # non-adjacent runs (A and C) are never paired: the same author may return
        self.assertFalse(any(a.startswith("a") and b.startswith("c") for a, b in negatives))

    def test_grouped_contains_every_consecutive_pair(self):
        grouped = {(p.text1, p.text2, p.change) for p in grouped_pairs(DOC)}
        for p in consecutive_pairs(DOC):
            self.assertIn((p.text1, p.text2, p.change), grouped)

    def test_frame_round_trip(self):
        pairs = build_pairs([DOC], "grouped")
        self.assertEqual(frame_to_pairs(pairs_frame(pairs)), pairs)

    def test_requires_truth(self):
        with self.assertRaises(ValueError):
            consecutive_pairs(Document("1", ["x", "y"]))


class MetricTests(unittest.TestCase):
    def test_macro_f1(self):
        y_true = [0, 0, 1, 1, 1, 1]
        y_pred = [0, 1, 1, 1, 1, 0]
        per_class = f1_per_class(y_true, y_pred)
        self.assertAlmostEqual(per_class[0], 0.5)
        self.assertAlmostEqual(per_class[1], 0.75)
        self.assertAlmostEqual(macro_f1(y_true, y_pred), 0.625)

    def test_evaluate_with_missing_predictions(self):
        docs = [Document("1", ["a", "b", "c"], [0, 1]), Document("2", ["a", "b"], [1])]
        perfect = evaluate(docs, {"1": [0, 1], "2": [1]})
        self.assertAlmostEqual(perfect["f1_macro"], 1.0)
        self.assertEqual(perfect["coverage"], 1.0)
        partial = evaluate(docs, {"1": [0, 1], "2": [1, 1]})  # wrong length -> fallback "change"
        self.assertEqual(partial["coverage"], 0.5)
        self.assertAlmostEqual(partial["f1_macro"], 1.0)
        missing = evaluate(docs, {})
        self.assertAlmostEqual(missing["f1_same"], 0.0)


if __name__ == "__main__":
    unittest.main()
