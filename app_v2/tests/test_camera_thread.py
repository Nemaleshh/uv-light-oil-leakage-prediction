import unittest
import numpy as np
import cv2
from core.camera_thread import CameraThread

class TestCameraThread(unittest.TestCase):
    def setUp(self):
        # We don't start the thread in these tests to avoid locking up USB resources.
        # We just test the logic functions.
        self.cam_thread = CameraThread(cam_index=-1)

    def test_is_uv_on_black_frame(self):
        """Test UV detection on a completely black frame (no UV)."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.assertFalse(self.cam_thread._is_uv_on(frame))

    def test_is_uv_on_white_frame(self):
        """Test UV detection on a completely white frame (no specific hue)."""
        frame = np.ones((720, 1280, 3), dtype=np.uint8) * 255
        self.assertFalse(self.cam_thread._is_uv_on(frame))

    def test_is_uv_on_violet_frame(self):
        """Test UV detection on a frame filled with the target UV hue."""
        # Create a purple frame in BGR
        # OpenCV HSV for purple is roughly H=140, S=200, V=200
        # Let's create an HSV frame and convert to BGR
        hsv_frame = np.zeros((120, 160, 3), dtype=np.uint8)
        hsv_frame[..., 0] = 145 # Hue
        hsv_frame[..., 1] = 200 # Saturation
        hsv_frame[..., 2] = 200 # Value
        bgr_frame = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2BGR)
        
        # It should trigger the UV on detection since it's 100% purple
        self.assertTrue(self.cam_thread._is_uv_on(bgr_frame))

    def test_is_uv_on_bad_input(self):
        """Test robustness against weird shapes or types."""
        # Should catch exceptions and return False
        self.assertFalse(self.cam_thread._is_uv_on(None))
        
        # Extremely small frame
        small_frame = np.zeros((2, 2, 3), dtype=np.uint8)
        self.assertFalse(self.cam_thread._is_uv_on(small_frame))

if __name__ == "__main__":
    unittest.main()
