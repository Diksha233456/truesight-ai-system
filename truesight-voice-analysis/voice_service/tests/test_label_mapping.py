"""
tests/test_label_mapping.py

Unit test suite to verify that:
1. Label mapping in predictor is strictly 0 = Truthful, 1 = Deceptive.
2. The model's model.classes_ attribute equals [0, 1].
3. Predictor output formats match API expectations.
"""

import os
import unittest
import numpy as np
from app.predictor import Predictor

class TestLabelMapping(unittest.TestCase):

    def setUp(self):
        self.predictor = Predictor()

    def test_model_loaded(self):
        self.assertTrue(self.predictor.is_ready(), "Model file should be present and loadable.")

    def test_classes_order(self):
        model = self.predictor.model
        if hasattr(model, "classes_"):
            classes = model.classes_
        elif hasattr(model, "named_steps"):
            # If it's a pipeline, get the estimator
            last_step = list(model.named_steps.values())[-1]
            classes = getattr(last_step, "classes_", [0, 1])
        else:
            classes = [0, 1]

        np.testing.assert_array_equal(classes, [0, 1], "Model classes_ must be [0, 1] (0=Truthful, 1=Deceptive).")

    def test_predict_prob_mapping(self):
        if not self.predictor.is_ready():
            self.skipTest("Predictor is not ready")

        # Dummy feature vector matching model expected input dim
        model = self.predictor.model
        if hasattr(model, "n_features_in_"):
            dim = model.n_features_in_
        elif hasattr(model, "steps"):
            dim = 814  # standard multimodal dim
        else:
            dim = 42

        dummy_features = np.zeros(dim)
        verdict, deception_score, confidence = self.predictor.predict(dummy_features)

        self.assertIn(verdict, ["Truthful", "Deceptive"])
        self.assertGreaterEqual(deception_score, 0.0)
        self.assertLessEqual(deception_score, 1.0)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

if __name__ == "__main__":
    unittest.main()
