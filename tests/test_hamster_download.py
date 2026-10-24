"""Exercise transfer failure and range-integrity gates without network access."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("bad_range", [False, True])
def test_download_retries_transient_errors_and_rejects_bad_ranges(tmp_path, bad_range):
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "scripts").mkdir()
    script = tmp_path / "scripts/fetch_hamster_ranges.py"
    script.write_bytes((root / "scripts/fetch_hamster_ranges.py").read_bytes())
    (tmp_path / ".local").mkdir()
    evidence = tmp_path / "evidence/hamster"
    evidence.mkdir(parents=True)
    payload = b"official-test-weight"
    manifest = {"repo": "test/fixture", "revision": "fixed", "files": [
        {"rfilename": "model.safetensors", "size": len(payload),
         "lfs": {"sha256": hashlib.sha256(payload).hexdigest()}}]}
    (evidence / "official_manifest.json").write_text(json.dumps(manifest))
    binaries = tmp_path / "bin"
    binaries.mkdir()
    curl = binaries / "curl"
    curl.write_text(f"#!{sys.executable}\n" + r'''
import os
import sys
from pathlib import Path
args = sys.argv[1:]
state = Path(os.environ["TEST_DOWNLOAD_STATE"])
if not state.exists():
    state.write_text("first request failed")
    sys.exit(22)
payload = b"official-test-weight"
Path(args[args.index("-o")+1]).write_bytes(payload)
start, end = map(int, args[args.index("--range")+1].split("-"))
if os.environ["TEST_BAD_RANGE"] == "1":
    end += 1
Path(args[args.index("-D")+1]).write_text(
    f"HTTP/1.1 206 Partial Content\nContent-Range: bytes {start}-{end}/{len(payload)}\n")
''')
    curl.chmod(0o755)
    env = {**os.environ, "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
           "TEST_DOWNLOAD_STATE": str(tmp_path / "attempted"),
           "TEST_BAD_RANGE": str(int(bad_range))}
    result = subprocess.run([sys.executable, str(script)], env=env, text=True,
                            capture_output=True, timeout=10)
    output = tmp_path / "models/hamster/model.safetensors"
    assert '"download_error": "CalledProcessError"' in result.stdout
    if bad_range:
        assert result.returncode != 0
        assert not output.exists()
        assert "chunks failed after three bounded rounds" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        assert output.read_bytes() == payload
        assert "Verified model.safetensors" in result.stdout
