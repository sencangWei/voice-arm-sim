"""CPU-only language-boundary tests; no model loading or motion."""
import io
import json
import unittest
from unittest.mock import patch

import model_service
from vla.run import translate_prompt, voice_prompt


class VoiceTests(unittest.TestCase):
    def test_translation_rejects_invalid_payload_before_model_access(self):
        for payload in (None, [], {}, {"text": " "}, {"text": 1}, {"text": "a" * 501}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                model_service.translate(payload)

    def test_chinese_text_uses_real_translation_endpoint(self):
        response = io.BytesIO(json.dumps({"text": "put the bowl on the plate"}).encode())
        with patch("urllib.request.urlopen", return_value=response) as request:
            self.assertEqual(translate_prompt("帮我把碗放到盘子上"), "put the bowl on the plate")
        self.assertEqual(json.loads(request.call_args.args[0].data), {"text": "帮我把碗放到盘子上"})

    def test_english_and_empty_text_do_not_call_models(self):
        with patch("urllib.request.urlopen") as request:
            self.assertEqual(translate_prompt(" pick up the cup "), "pick up the cup")
            with self.assertRaises(ValueError):
                translate_prompt(" ")
            request.assert_not_called()

    def test_bad_translation_never_becomes_a_policy_prompt(self):
        for text in ("", "还是中文", None):
            with self.subTest(text=text), patch("urllib.request.urlopen", return_value=io.BytesIO(
                    json.dumps({"text": text}).encode())), self.assertRaises(ValueError):
                translate_prompt("拿起碗")

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
