"""Fetch generation-pinned official pi05 LIBERO objects with integrity checks."""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import quote
from urllib.request import Request, urlopen

import google_crc32c

PREFIX = "checkpoints/pi05_libero/"
LIST_URL = ("https://storage.googleapis.com/storage/v1/b/openpi-assets/o?"
            "prefix=checkpoints/pi05_libero/&maxResults=1000&"
            "fields=items(name,size,generation,md5Hash,crc32c),nextPageToken")


def verify(path, item):
    sha = hashlib.sha256()
    crc = google_crc32c.Checksum()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            sha.update(chunk)
            crc.update(chunk)
    if path.stat().st_size != int(item["size"]):
        raise ValueError(f"Size mismatch: {path}")
    if base64.b64encode(crc.digest()).decode() != item["crc32c"]:
        raise ValueError(f"CRC32C mismatch: {path}")
    return sha.hexdigest()


def fetch(item, destination):
    relative = item["name"].removeprefix(PREFIX)
    path = destination / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    size = int(item["size"])
    if not path.exists():
        url = ("https://storage.googleapis.com/storage/v1/b/openpi-assets/o/"
               + quote(item["name"], safe="")
               + "?alt=media&generation=" + item["generation"])
        # Some gateways buffer multi-GB responses and fail before sending headers.
        # Bounded ranges avoid that failure and permit interrupted-run recovery.
        block_size = 16 * 1024 * 1024
        def fetch_block(offset):
            end = min(offset + block_size, size) - 1
            block = path.with_name(path.name + f".part-{offset}")
            expected = end - offset + 1
            if block.exists() and block.stat().st_size == expected:
                return block
            for attempt in range(5):
                try:
                    request = Request(url + f"&range_start={offset}",
                                      headers={"Range": f"bytes={offset}-{end}"})
                    with urlopen(request, timeout=60) as response:
                        if response.status != 206 or response.headers.get("Content-Range") != f"bytes {offset}-{end}/{size}":
                            raise ValueError("Server did not honor exact download range")
                        with block.open("wb") as stream:
                            while chunk := response.read(1024 * 1024):
                                stream.write(chunk)
                    if block.stat().st_size != expected:
                        raise OSError("Incomplete range response")
                    return block
                except OSError as exc:
                    if attempt == 4:
                        raise
                    print(f"Retry {relative}@{offset}: {exc}", flush=True)
                    time.sleep(2)
        with ThreadPoolExecutor(max_workers=8) as blocks:
            pieces = list(blocks.map(fetch_block, range(0, size, block_size)))
        with partial.open("wb") as stream:
            for piece in pieces:
                with piece.open("rb") as source:
                    while chunk := source.read(8 * 1024 * 1024):
                        stream.write(chunk)
        try:
            checksum = verify(partial, item)
        except ValueError:
            # A full-length cached range may be corrupt. Discard this object's
            # unverified download cache so the next run actually repairs it.
            partial.unlink(missing_ok=True)
            for piece in pieces:
                piece.unlink(missing_ok=True)
            raise
        partial.rename(path)
        for piece in pieces:
            piece.unlink()
    else:
        checksum = verify(path, item)
    print(f"Verified {relative}: {size:,} bytes", flush=True)
    return {**item, "relative_path": relative, "sha256": checksum}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/pi05_libero"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    inventory = args.output / "remote-objects.json"
    if inventory.exists():
        listing = json.loads(inventory.read_text())
    else:
        with urlopen(LIST_URL, timeout=30) as response:
            listing = json.load(response)
        if listing.get("nextPageToken") or not listing.get("items"):
            raise ValueError("Incomplete official checkpoint listing")
        inventory.write_text(json.dumps(listing, indent=2) + "\n")
    items = listing["items"]
    print(f"Official pi05_libero: {len(items)} objects, {sum(int(x['size']) for x in items):,} bytes", flush=True)
    with ThreadPoolExecutor(max_workers=8) as executor:
        records = list(executor.map(lambda item: fetch(item, args.output), items))
    manifest = {"source": "gs://openpi-assets/checkpoints/pi05_libero", "objects": records}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("COMPLETE: all objects verified against official CRC32C; SHA256 recorded", flush=True)


if __name__ == "__main__":
    main()
