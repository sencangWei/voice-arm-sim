"""CPU-only language-boundary tests; no model loading or motion."""
import io
import json
import unittest
from unittest.mock import patch

import model_service
from vla.run import voice_prompt


class VoiceTests(unittest.TestCase):
    def test_translation_rejects_invalid_payload_before_model_access(self):
        for payload in (None, [], {}, {"text": " "}, {"text": 1}, {"text": "a" * 501}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                model_service.translate(payload)

    def test_voice_only_requests_asr_and_translation(self):
        requests = []
        def respond(request, timeout):
            requests.append(request)
            result = {"text": "把杯子拿起来"} if len(requests) == 1 else {
                "original_text": "把杯子拿起来", "text": "pick up the cup"}
            return io.BytesIO(json.dumps(result).encode())
        with patch("urllib.request.urlopen", side_effect=respond):
            self.assertEqual(voice_prompt(b"wav"), ("把杯子拿起来", "pick up the cup"))
        self.assertEqual([r.full_url for r in requests], [
            "http://127.0.0.1:8911/transcribe", "http://127.0.0.1:8911/translate"])
        self.assertEqual(json.loads(requests[1].data), {"text": "把杯子拿起来"})


if __name__ == "__main__":
    unittest.main()
