import unittest
import os
import tempfile
from core.report_generator import generate_car_report, generate_summary_report, _image_to_base64

class TestReportGenerator(unittest.TestCase):
    def setUp(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.test_dir = tempfile.TemporaryDirectory()
        self.reports_dir = self.test_dir.name

    def tearDown(self):
        self.test_dir.cleanup()

    def test_image_to_base64_invalid_path(self):
        """Test encoding fails gracefully with invalid path."""
        res = _image_to_base64("this_does_not_exist.jpg")
        self.assertIsNone(res)

    def test_image_to_base64_valid(self):
        """Test encoding works on a small dummy image."""
        img_path = os.path.join(self.reports_dir, "dummy.jpg")
        # Write dummy binary data
        with open(img_path, "wb") as f:
            f.write(b"\xff\xd8\xff\xe0")
        
        res = _image_to_base64(img_path)
        self.assertIsNotNone(res)
        self.assertTrue(res.startswith("data:image/jpeg;base64,"))

    def test_generate_car_report(self):
        """Test HTML report generation creates a file without crashing, even with weird VINs."""
        vin = "W!E#I@R$D-VIN*"
        date_str = "2024-05-01"
        time_str = "12:00:00"
        status = "OIL LEAK"
        verification = "System Match"
        
        out_path = generate_car_report(
            vin, date_str, time_str, status, verification, 
            photo_path="nonexistent.jpg", 
            reports_dir=self.reports_dir, 
            base_dir=self.base_dir
        )
        
        self.assertTrue(os.path.exists(out_path))
        with open(out_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        self.assertIn(vin, content)
        self.assertIn("No Photo Available", content)

    def test_generate_summary_report_missing_csv(self):
        """Test summary generator handles missing CSV."""
        out_path = generate_summary_report(
            csv_path="no_csv_here.csv",
            reports_dir=self.reports_dir,
            base_dir=self.base_dir
        )
        self.assertIsNone(out_path)

    def test_generate_summary_report_valid_csv(self):
        """Test summary generation with a mock CSV."""
        csv_path = os.path.join(self.reports_dir, "mock_log.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("vin_id,timestamp,auto_result,manual_confirm\n")
            f.write("VIN123,2024-01-01 10:00:00,NO LEAK,No Leak\n")
            f.write("VIN456,2024-01-01 10:05:00,OIL LEAK,Engine Oil\n")
            f.write("VIN789,2024-01-01 10:10:00,NO LEAK,Engine Oil\n")

        out_path = generate_summary_report(
            csv_path=csv_path,
            reports_dir=self.reports_dir,
            base_dir=self.base_dir
        )
        
        self.assertIsNotNone(out_path)
        self.assertTrue(os.path.exists(out_path))
        
        with open(out_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        self.assertIn("VIN123", content)
        # 2 matches (VIN123, VIN456), 1 mismatch (VIN789)
        self.assertIn('<div class="value">2</div>', content) # system matches
        self.assertIn('<div class="value">1</div>', content) # system mismatches

if __name__ == "__main__":
    unittest.main()
