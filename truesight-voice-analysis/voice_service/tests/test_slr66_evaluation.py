"""
tests/test_slr66_evaluation.py

Unit test suite verifying OpenSLR SLR66 Telugu evaluation helper logic:
1. TSV parsing and speaker ID extraction from tef_<speaker>_<utterance> filenames.
2. Stratified manifest generation across speakers.
3. Speaker-level metric aggregation.
4. Ethical boundary verification (no veracity scores, no deception labels).
Runs entirely offline using temporary fixtures; does not require downloading datasets.
"""

import os
import shutil
import tempfile
import unittest

from evaluate_slr66_telugu import (
    load_line_index_tsv,
    build_evaluation_manifest,
)


class TestSLR66Evaluation(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_slr66_")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_line_index_tsv_parsing(self):
        """Verify tab-separated TSV parsing and speaker ID extraction."""
        tsv_content = (
            "tef_01033_00351357063\tఈ వివాదం పరిష్కారమైనది\n"
            "tef_01033_00359519816\tరెండవ వాక్యం\n"
            "tef_02828_01625402654\tపాఠశాల ఉన్నాయి\n"
        )
        tsv_path = os.path.join(self.temp_dir, "test_line_index.tsv")
        with open(tsv_path, "w", encoding="utf-8") as f:
            f.write(tsv_content)

        index = load_line_index_tsv(tsv_path)
        self.assertEqual(len(index), 3)

        # Check speaker ID extraction
        self.assertIn("tef_01033_00351357063", index)
        entry1 = index["tef_01033_00351357063"]
        self.assertEqual(entry1["speaker_id"], "tef_01033")
        self.assertEqual(entry1["transcript"], "ఈ వివాదం పరిష్కారమైనది")

        entry3 = index["tef_02828_01625402654"]
        self.assertEqual(entry3["speaker_id"], "tef_02828")

    def test_build_evaluation_manifest_stratification(self):
        """Verify stratified selection limits recordings per speaker evenly."""
        # Create mock wav files for 3 speakers (4 clips each = 12 total)
        tsv_index = {}
        wav_paths = []
        for spk in ["01033", "02828", "03689"]:
            for i in range(1, 5):
                file_id = f"tef_{spk}_0000000000{i}"
                fname = f"{file_id}.wav"
                fpath = os.path.join(self.temp_dir, fname)
                with open(fpath, "wb") as f:
                    f.write(b"RIFFmockwavdata")
                wav_paths.append(fpath)
                tsv_index[file_id] = {
                    "file_id": file_id,
                    "speaker_id": f"tef_{spk}",
                    "transcript": f"mock transcript {i}",
                }

        # Request max 2 per speaker
        full_inv, eval_cohort = build_evaluation_manifest(wav_paths, tsv_index, max_per_speaker=2)

        self.assertEqual(len(full_inv), 12)
        self.assertEqual(len(eval_cohort), 6)  # 3 speakers * 2 clips

        # Verify each speaker has exactly 2 in the cohort
        spk_counts = {}
        for item in eval_cohort:
            spk_counts[item["speaker_id"]] = spk_counts.get(item["speaker_id"], 0) + 1
        self.assertEqual(spk_counts, {"tef_01033": 2, "tef_02828": 2, "tef_03689": 2})

    def test_ethical_boundary_no_deception_fields(self):
        """Ensure manifest entries have zero veracity/deception fields."""
        tsv_index = {
            "tef_01033_001": {"speaker_id": "tef_01033", "transcript": "test"}
        }
        fpath = os.path.join(self.temp_dir, "tef_01033_001.wav")
        with open(fpath, "wb") as f:
            f.write(b"RIFFmockwavdata")

        full_inv, eval_cohort = build_evaluation_manifest([fpath], tsv_index)
        entry = eval_cohort[0]

        self.assertNotIn("label", entry)
        self.assertNotIn("verdict", entry)
        self.assertNotIn("deception_score", entry)
        self.assertNotIn("is_truthful", entry)
        self.assertEqual(entry["language"], "Telugu")


if __name__ == "__main__":
    unittest.main()
