import unittest
from experiment_v2 import calculate_advanced_metrics

class TestExperimentMetrics(unittest.TestCase):
    def test_calculate_advanced_metrics(self):
        gt = {"1.1.1", "1.3.1"}
        preds = {"1.1.1"}
        metrics = calculate_advanced_metrics(gt, preds)
        
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertGreater(metrics["f1_score"], 0)

if __name__ == "__main__":
    unittest.main()
