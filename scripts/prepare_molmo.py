"""Download a pinned public model and verify each official LFS SHA256."""
import argparse
import hashlib
import json
from urllib.request import urlopen

from huggingface_hub import snapshot_download

from vla.molmo import REPO, REVISION, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    output = ROOT / "models/molmoact2/checkpoint"
    if not args.verify_only:
        snapshot_download(REPO, revision=REVISION, local_dir=output, max_workers=5)
    with urlopen(f"https://huggingface.co/api/models/{REPO}/tree/{REVISION}?recursive=true", timeout=60) as response:
        entries = json.load(response)
    files = []
    for entry in entries:
        if entry["type"] != "file":
            continue
        path = output / entry["path"]
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        expected = entry.get("lfs", {}).get("oid")
        if path.stat().st_size != entry["size"] or (expected and digest != expected):
            raise ValueError(f"Official checksum/size mismatch: {path.name}")
        files.append(dict(path=entry["path"], size=entry["size"], sha256=digest,
                          official_lfs_sha256=expected))
    manifest = dict(repo=REPO, revision=REVISION, files=files)
    (output / "verified-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    evidence = ROOT / "evidence/molmoact2"
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / "checkpoint.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Verified {len(files)} official files: {output}", flush=True)


if __name__ == "__main__":
    main()
