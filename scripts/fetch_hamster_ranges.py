"""Local curl fallback for broken Xet connections; pinned SHA256 verification."""
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'models/hamster'
CHUNKS = ROOT / '.local/hamster-download-chunks'
CHUNKS.mkdir(exist_ok=True)
BLOCK = 32 * 1024**2


def curl(url, path, headers=None, byte_range=None):
    command = ['curl', '-L', '-s', '--fail', '--connect-timeout', '20',
               '--max-time', '180', '--retry', '2', '--retry-all-errors',
               '--retry-delay', '2', '-o', str(path)]
    if headers is not None:
        command += ['-D', str(headers)]
    if byte_range is not None:
        command += ['--range', byte_range]
    subprocess.run(command + [url], check=True)


manifest = json.loads((ROOT / 'evidence/hamster/official_manifest.json').read_text())
REPO, REV = manifest['repo'], manifest['revision']
entries = [{**e, 'path': e['rfilename']} for e in manifest['files']]

def get_chunk(job):
    name, start, end, total = job
    dest = CHUNKS / f'{name}.{start:012d}.part'
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == end - start + 1:
        return dest.stat().st_size
    temp, header = dest.with_suffix('.tmp'), dest.with_suffix('.headers')
    url = f'https://huggingface.co/{REPO}/resolve/{REV}/{name}?segment={start}'
    curl(url, temp, header, f'{start}-{end}')
    ranges = re.findall(r'content-range:\s*bytes\s+(\d+)-(\d+)/(\d+)',
                        header.read_text().lower())
    if (not ranges or tuple(map(int, ranges[-1])) != (start, end, total)
            or temp.stat().st_size != end - start + 1):
        raise RuntimeError(f'Bad content range/length: {name} at {start}')
    temp.replace(dest)
    header.unlink()
    return dest.stat().st_size


jobs = []
small_files = []
weights = []
for entry in entries:
    name, size = entry['path'], entry['size']
    dest = DEST / name
    if name.endswith('.safetensors'):
        if dest.exists() and dest.stat().st_size == size:
            with dest.open('rb') as source:
                digest = hashlib.file_digest(source, 'sha256').hexdigest()
            if digest == entry['lfs']['sha256']:
                print('Already verified', name, flush=True)
                continue
        weights.append(entry)
        jobs.extend((name, start, min(start + BLOCK, size) - 1, size)
                    for start in range(0, size, BLOCK))
    elif not dest.exists() or dest.stat().st_size != size:
        small_files.append((name, dest))

started, finished = time.monotonic(), 0
with ThreadPoolExecutor(max_workers=24) as pool:
    futures = [pool.submit(get_chunk, job) for job in jobs]
    for future in as_completed(futures):
        finished += future.result()
        print(json.dumps({'chunks_done_bytes': finished,
                          'total_bytes': sum(x['size'] for x in weights),
                          'elapsed_s': round(time.monotonic() - started, 1)}), flush=True)

for entry in weights:
    name = entry['path']
    dest = DEST / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix('.assembling')
    with temp.open('wb') as output:
        for start in range(0, entry['size'], BLOCK):
            with (CHUNKS / f'{name}.{start:012d}.part').open('rb') as source:
                shutil.copyfileobj(source, output, 8 * 1024**2)
        output.flush()
        os.fsync(output.fileno())
    with temp.open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    if digest != entry['lfs']['sha256']:
        raise RuntimeError(f'Official SHA256 mismatch: {name}')
    temp.replace(dest)
    print('Verified', name, digest, flush=True)

for name, dest in small_files:
    dest.parent.mkdir(parents=True, exist_ok=True)
    curl(f'https://huggingface.co/{REPO}/resolve/{REV}/{name}?download=true', dest.with_suffix('.download'))
    dest.with_suffix('.download').replace(dest)
