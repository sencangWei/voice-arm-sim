"""Fetch clean, pinned upstream sources into this project's ignored .local folder."""

import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PINS = {
    "openpi": "15a9616a00943ada6c20a0f158e3adb39df2ccac",
    "libero": "f78abd68ee283de9f9be3c8f7e2a9ad60246e95c",
}


def main():
    destination = ROOT / ".local/openpi"
    if destination.exists():
        if json.loads((destination / "SNAPSHOT.json").read_text()) != PINS:
            raise ValueError("Existing source differs; preserve and inspect it")
        print("Pinned source snapshot already present")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="voice-arm-upstream-", dir=destination.parent) as temporary:
        staged = Path(temporary) / "snapshot"
        for name, repo, output in (
            ("openpi", "https://github.com/Physical-Intelligence/openpi.git", staged),
            ("libero", "https://github.com/Lifelong-Robot-Learning/LIBERO.git", staged / "third_party/libero"),
        ):
            checkout = Path(temporary) / name
            subprocess.run(["git", "init", "-q", str(checkout)], check=True)
            subprocess.run(["git", "-C", str(checkout), "fetch", "--depth=1", repo, PINS[name]], check=True)
            archive = subprocess.check_output(["git", "-C", str(checkout), "archive", "FETCH_HEAD"])
            output.mkdir(parents=True, exist_ok=True)
            with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
                stream.extractall(output, filter="data")
        (staged / "SNAPSHOT.json").write_text(json.dumps(PINS, indent=2) + "\n")
        shutil.move(staged, destination)
    print("Created independent pinned openpi + LIBERO source snapshot")


if __name__ == "__main__":
    main()
