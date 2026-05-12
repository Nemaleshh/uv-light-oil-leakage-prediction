import unittest
import numpy as np
import os
import cv2
from core.pipeline import LeakPipeline

class TestLeakPipeline(unittest.TestCase):
    def setUp(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_path = os.path.join(base_dir, "models", "best.pt")
        self.clf_path = os.path.join(base_dir, "models", "gb_classifier.pkl")
        self.pipeline = LeakPipeline(self.model_path, self.clf_path)

    def test_load_models_success(self):
        """Test models load successfully when paths are valid."""
        if os.path.exists(self.model_path) and os.path.exists(self.clf_path):
            try:
                self.pipeline.load()
                self.assertTrue(self.pipeline.is_loaded)
            except Exception as e:
                self.fail(f"load() raised Exception unexpectedly: {e}")
        else:
            self.skipTest("Models not found on disk, skipping load test.")

    def test_load_models_failure(self):
        """Test model load fails with FileNotFoundError when paths are invalid."""
        bad_pipeline = LeakPipeline("invalid_model.pt", "invalid_clf.pkl")
        with self.assertRaises(FileNotFoundError):
            bad_pipeline.load()

    def test_run_not_loaded(self):
        """Test run() before load() returns an error dict."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        res = self.pipeline.run(frame)
        self.assertEqual(res["result"], "ERROR")
        self.assertIn("not loaded", res["error"])

    def test_run_valid_empty_frame(self):
        """Test inference handles an empty/black frame without crashing."""
        if not os.path.exists(self.model_path) or not os.path.exists(self.clf_path):
            self.skipTest("Models missing.")
            
        self.pipeline.load()
        # A completely black frame should not find any target
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        res = self.pipeline.run(frame)
        self.assertEqual(res["result"], "NO_PART")
        self.assertIsNone(res["error"])

    def test_run_invalid_type(self):
        """Test pipeline robustness against bad input types (loopholes)."""
        if not os.path.exists(self.model_path) or not os.path.exists(self.clf_path):
            self.skipTest("Models missing.")
            
        self.pipeline.load()
        # Pass None instead of np.ndarray
        res = self.pipeline.run(None)
        # It shouldn't crash. It might return ERROR or NO_PART depending on YOLO internals
        self.assertIn(res["result"], ["ERROR", "NO_PART"])

    def test_extract_features_shape(self):
        """Test feature extraction yields a vector of length 2137."""
        # This tests the extraction logic handles a small numpy array correctly
        dummy_crop = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        feats = self.pipeline._extract_features(dummy_crop)
        # Check that it returns a 1D numpy array
        self.assertEqual(len(feats.shape), 1)
        # Should be exactly 2137 features based on test_pipeline.ipynb specification
        self.assertEqual(feats.shape[0], 2137)

if __name__ == "__main__":
    unittest.main()
