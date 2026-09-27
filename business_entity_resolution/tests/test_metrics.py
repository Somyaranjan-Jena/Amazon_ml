import unittest
from business_entity_resolution.src.metrics import compute_f05_single, evaluate_predictions

class TestMetrics(unittest.TestCase):
    def test_readme_example(self):
        # From README: S1-00001 predicted [S2-00047, S2-00193, S3-00812]
        # Ground truth [S2-00047, S3-00812]
        gt = {"S2-00047", "S3-00812"}
        pred = {"S2-00047", "S2-00193", "S3-00812"}
        f05, prec, rec = compute_f05_single(gt, pred)
        self.assertAlmostEqual(prec, 2.0 / 3.0, places=4)
        self.assertAlmostEqual(rec, 1.0, places=4)
        self.assertAlmostEqual(f05, 0.7142857, places=4)

    def test_singleton_correct(self):
        # Empty GT and Empty Pred -> 1.0
        f05, prec, rec = compute_f05_single(set(), set())
        self.assertEqual(f05, 1.0)

    def test_singleton_false_merge(self):
        # Empty GT and Non-empty Pred -> 0.0
        f05, prec, rec = compute_f05_single(set(), {"S2-123"})
        self.assertEqual(f05, 0.0)

    def test_missed_matches(self):
        # Non-empty GT and Empty Pred -> 0.0
        f05, prec, rec = compute_f05_single({"S2-123"}, set())
        self.assertEqual(f05, 0.0)

    def test_macro_evaluation(self):
        gt = {
            "S1-1": {"S2-A", "S3-B"},
            "S1-2": set(), # singleton
            "S1-3": {"S2-C"}
        }
        pred = {
            "S1-1": {"S2-A", "S3-B"}, # Perfect: 1.0
            "S1-2": set(),            # Perfect singleton: 1.0
            "S1-3": {"S2-D"}          # Complete mismatch: 0.0
        }
        res = evaluate_predictions(gt, pred)
        self.assertAlmostEqual(res["macro_f05"], 2.0 / 3.0, places=4)
        self.assertEqual(res["singleton_accuracy"], 1.0)

if __name__ == "__main__":
    unittest.main()
