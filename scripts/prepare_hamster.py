"""Download the pinned official HAMSTER checkpoint and verify its LFS hashes."""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    evidence = ROOT / "evidence/hamster"
    manifest = json.loads((evidence / "official_manifest.json").read_text())
    output = ROOT / "models/hamster"
    if not args.verify_only:
        snapshot_download(manifest["repo"], revision=manifest["revision"],
                          allow_patterns=[f["rfilename"] for f in manifest["files"]],
                          local_dir=output, max_workers=3)
    verified = []
    for entry in manifest["files"]:
        path = output / entry["rfilename"]
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        expected = entry.get("lfs", {}).get("sha256")
        if path.stat().st_size != entry["size"] or (expected and expected != digest):
            raise ValueError(f"Official checksum/size mismatch: {path}")
        verified.append({"path": entry["rfilename"], "sha256": digest, "size": entry["size"]})
    (evidence / "verified-checkpoint.json").write_text(json.dumps({
        "repo": manifest["repo"], "revision": manifest["revision"], "files": verified}, indent=2) + "\n")
    print(f"Verified {len(verified)} files: {output / manifest['subfolder']}", flush=True)


if __name__ == "__main__":
    main()
