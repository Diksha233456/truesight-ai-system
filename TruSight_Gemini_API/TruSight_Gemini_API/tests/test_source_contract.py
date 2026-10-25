"""Offline contract tests for the API-migrated TruSight project."""
import ast
import os
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class SourceContractTests(unittest.TestCase):
    def test_all_python_files_parse(self):
        for path in ROOT.glob("*.py"):
            with self.subTest(path=path.name):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_api_key_is_environment_configured(self):
        client_source = (ROOT / "gemini_client.py").read_text(encoding="utf-8")
        env_example = (ROOT / ".env.example").read_text(encoding="utf-8")
        self.assertIn('os.getenv("GEMINI_API_KEY")', client_source)
        self.assertIn("GEMINI_API_KEY=replace_with_your_key", env_example)
        self.assertNotRegex(client_source, r"api_key\s*=\s*['\"][^'\"]{8,}['\"]")

    def test_local_inference_dependencies_removed(self):
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
        for dependency in ("torch", "transformers", "sentence-transformers", "openai-whisper", "ollama", "faiss"):
            with self.subTest(dependency=dependency):
                self.assertNotIn(dependency, requirements)

    def test_live_audio_and_pdf_routes_preserved(self):
        main = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn('@app.post("/case")', main)
        self.assertIn('@app.websocket("/ws/audio/{case_id}")', main)
        self.assertIn("send_realtime_input", main)
        self.assertIn("interim_input_transcription", main)

    def test_frontend_javascript_has_no_stray_backtick_line(self):
        main = (ROOT / "main.py").read_text(encoding="utf-8")
        match = re.search(r"<script>(.*?)</script>", main, re.S)
        self.assertIsNotNone(match)
        self.assertNotRegex(match.group(1), r"(?m)^\s*`\s*$")


if __name__ == "__main__":
    unittest.main()
