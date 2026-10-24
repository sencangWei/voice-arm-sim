"""Integrity and controlled-ablation regressions, without model loading."""
import base64
import hashlib
from pathlib import Path
import tempfile
import unittest
import google_crc32c
import numpy as np
from vla.fetch_checkpoint import verify
from vla.serve import ProbePolicy


class ArtifactTests(unittest.TestCase):
    def test_corrupted_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blob"
            expected = b"official downloaded tensor bytes"
            item = {"size": str(len(expected)), "crc32c": base64.b64encode(google_crc32c.Checksum(expected).digest()).decode()}
            path.write_bytes(expected)
            self.assertEqual(verify(path, item), hashlib.sha256(expected).hexdigest())
            path.write_bytes(b"X" + expected[1:])
            with self.assertRaisesRegex(ValueError, "CRC32C"):
                verify(path, item)

    def test_language_probe_keeps_identical_noise(self):
        class Recorder:
            def infer(self, obs, **kwargs):
                return dict(obs=obs, **kwargs)
        proxy = ProbePolicy(Recorder(), 10, 32)
        first = {"prompt": "move bowl", "_probe_noise_seed": 42}
        a = proxy.infer(first)
        b = proxy.infer({"prompt": "open drawer", "_probe_noise_seed": 42})
        np.testing.assert_array_equal(a["noise"], b["noise"])
        self.assertEqual(a["noise"].shape, (10, 32))
        self.assertNotIn("_probe_noise_seed", a["obs"])
        self.assertIn("_probe_noise_seed", first)
        self.assertNotIn("noise", proxy.infer({"prompt": "default policy"}))
        with self.assertRaises(ValueError):
            proxy.infer({"_probe_noise_seed": -1})
