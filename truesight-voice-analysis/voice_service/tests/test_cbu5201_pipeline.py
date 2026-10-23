"""
tests/test_cbu5201_pipeline.py

Unit test suite verifying CBU5201 dataset preparation, security guardrails,
label mapping, inventory generation, and evaluation metric computations.
Runs entirely offline without requiring fresh internet downloads.
"""

import csv
import io
import os
import shutil
import tempfile
import unittest
import zipfile

import numpy as np

from prepare_cbu5201 import (
    CBU5201PreparationError,
    safe_extract_zip,
    map_story_type_to_label,
    generate_manifest_and_inventory,
)
from evaluate_cbu5201 import calculate_binary_metrics


class TestCBU5201Pipeline(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_cbu5201_")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_map_story_type_to_label_valid(self):
        """Verify strict mapping of story types to binary integer labels."""
        self.assertEqual(map_story_type_to_label("True Story"), 0)
        self.assertEqual(map_story_type_to_label("true story"), 0)
        self.assertEqual(map_story_type_to_label("truthful"), 0)
        self.assertEqual(map_story_type_to_label("Deceptive Story"), 1)
        self.assertEqual(map_story_type_to_label("deceptive story"), 1)
        self.assertEqual(map_story_type_to_label("deceptive"), 1)

    def test_map_story_type_to_label_ambiguous(self):
        """Verify ambiguous, unknown, or blank labels return None and are never coerced."""
        self.assertIsNone(map_story_type_to_label(""))
        self.assertIsNone(map_story_type_to_label(None))
        self.assertIsNone(map_story_type_to_label("Uncertain"))
        self.assertIsNone(map_story_type_to_label("Unknown Story"))
        self.assertIsNone(map_story_type_to_label("Inconclusive"))

    def test_safe_extract_zip_prevents_path_traversal(self):
        """Security test: Zip Slip path traversal attempt must be blocked with error."""
        malicious_zip = os.path.join(self.temp_dir, "malicious.zip")
        with zipfile.ZipFile(malicious_zip, "w") as zf:
            # Add a file attempting to write outside target extraction directory
            zf.writestr("../../evil_escape.txt", b"malicious content")

        extract_target = os.path.join(self.temp_dir, "safe_target")
        with self.assertRaises(CBU5201PreparationError) as ctx:
            safe_extract_zip(malicious_zip, extract_target)

        self.assertIn("Zip Slip path traversal", str(ctx.exception))
        # Ensure the malicious file was NOT written outside
        escaped_file = os.path.abspath(os.path.join(extract_target, "../../evil_escape.txt"))
        self.assertFalse(os.path.exists(escaped_file))

    def test_safe_extract_zip_valid_archive(self):
        """Verify valid zip extraction works smoothly."""
        valid_zip = os.path.join(self.temp_dir, "valid.zip")
        with zipfile.ZipFile(valid_zip, "w") as zf:
            zf.writestr("subfolder/sample.txt", b"hello cbu5201")

        extract_target = os.path.join(self.temp_dir, "extracted_target")
        safe_extract_zip(valid_zip, extract_target)

        extracted_file = os.path.join(extract_target, "subfolder", "sample.txt")
        self.assertTrue(os.path.exists(extracted_file))
        with open(extracted_file, "rb") as f:
            self.assertEqual(f.read(), b"hello cbu5201")

    def test_safe_extract_nonexistent_or_corrupt_zip(self):
        """Verify proper error reporting on missing or corrupt zip files."""
        nonexistent = os.path.join(self.temp_dir, "nonexistent.zip")
        with self.assertRaises(CBU5201PreparationError):
            safe_extract_zip(nonexistent, self.temp_dir)

        corrupt_zip = os.path.join(self.temp_dir, "corrupt.zip")
        with open(corrupt_zip, "w") as f:
            f.write("not a zip file content")
        with self.assertRaises(CBU5201PreparationError):
            safe_extract_zip(corrupt_zip, self.temp_dir)

    def test_generate_manifest_and_inventory(self):
        """Verify manifest creation, label validation, and inventory accounting."""
        audio_dir = os.path.join(self.temp_dir, "audio")
        os.makedirs(audio_dir, exist_ok=True)

        # Create 2 valid dummy wav files on disk
        with open(os.path.join(audio_dir, "clip1.wav"), "wb") as f:
            f.write(b"RIFFdummydata")
        with open(os.path.join(audio_dir, "clip2.wav"), "wb") as f:
            f.write(b"RIFFdummydata")

        # Mock CSV containing:
        # clip1.wav -> True Story (exists)
        # clip2.wav -> Deceptive Story (exists)
        # clip3.wav -> True Story (missing on disk)
        # clip4.wav -> Unknown (ambiguous label)
        mock_csv_path = os.path.join(self.temp_dir, "mock_attributes.csv")
        with open(mock_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "Language", "Story_type"])
            writer.writerow(["clip1.wav", "English", "True Story"])
            writer.writerow(["clip2.wav", "Chinese", "Deceptive Story"])
            writer.writerow(["clip3.wav", "English", "True Story"])
            writer.writerow(["clip4.wav", "English", "Ambiguous"])

        manifest_out = os.path.join(self.temp_dir, "test_manifest.csv")
        inventory = generate_manifest_and_inventory(
            mock_csv_path,
            audio_dir,
            manifest_out,
            project_root=self.temp_dir,
        )

        self.assertEqual(inventory["total_metadata_rows"], 4)
        self.assertEqual(inventory["total_audio_files_found_on_disk"], 2)
        self.assertEqual(inventory["valid_labelled_samples"], 2)
        self.assertEqual(inventory["truthful_count"], 1)
        self.assertEqual(inventory["deceptive_count"], 1)
        self.assertEqual(inventory["missing_audio_files"], ["clip3.wav"])
        self.assertEqual(len(inventory["missing_or_ambiguous_labels"]), 1)
        self.assertEqual(inventory["missing_or_ambiguous_labels"][0]["filename"], "clip4.wav")
        self.assertFalse(inventory["speaker_identifier_present_in_metadata"])

        # Check generated manifest CSV rows
        with open(manifest_out, "r", encoding="utf-8") as f:
            manifest_rows = list(csv.DictReader(f))

        self.assertEqual(len(manifest_rows), 4)
        row1 = manifest_rows[0]
        self.assertEqual(row1["filename"], "clip1.wav")
        self.assertEqual(row1["label"], "0")
        self.assertEqual(row1["file_exists"], "True")
        self.assertEqual(row1["is_valid_labelled_sample"], "True")

        row3 = manifest_rows[2]
        self.assertEqual(row3["filename"], "clip3.wav")
        self.assertEqual(row3["file_exists"], "False")
        self.assertEqual(row3["is_valid_labelled_sample"], "False")

        row4 = manifest_rows[3]
        self.assertEqual(row4["filename"], "clip4.wav")
        self.assertEqual(row4["label"], "")
        self.assertEqual(row4["is_valid_labelled_sample"], "False")

    def test_calculate_binary_metrics(self):
        """Verify metric calculation: Accuracy, Balanced Accuracy, Macro F1, Confusion Matrix."""
        # y_true: [0, 0, 1, 1]
        # y_pred: [0, 1, 0, 1]
        # TN=1, FP=1, FN=1, TP=1
        metrics = calculate_binary_metrics([0, 0, 1, 1], [0, 1, 0, 1])

        self.assertEqual(metrics["sample_count"], 4)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["balanced_accuracy"], 0.5)
        self.assertEqual(metrics["confusion_matrix"]["tn"], 1)
        self.assertEqual(metrics["confusion_matrix"]["fp"], 1)
        self.assertEqual(metrics["confusion_matrix"]["fn"], 1)
        self.assertEqual(metrics["confusion_matrix"]["tp"], 1)


if __name__ == "__main__":
    unittest.main()
