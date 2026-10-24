"""Pair actual camera recordings by control index; never accelerate either side."""
import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--hybrid', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    paths = [args.baseline, args.hybrid]
    reports = [json.loads((path / 'result.json').read_text()) for path in paths]
    assert reports[0]['image_sha256'] == reports[1]['image_sha256']
    assert reports[0]['prompt'] == reports[1]['prompt']
    assert reports[0]['scheme'] == 'official pi05 policy-only matched control'
    readers = [imageio.get_reader(path / 'rollout.mp4') for path in paths]
    counts = [reader.count_frames() for reader in readers]
    assert all(reader.get_meta_data()['fps'] == 20 for reader in readers)
    assert counts == [report['control_steps'] + 1 for report in reports]
    font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 23)
    small = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 18)
    output = args.output / 'bottle-bowl-before-after.mp4'
    last = [None, None]
    with imageio.get_writer(output, fps=20, macro_block_size=16) as writer:
        for index in range(max(counts)):
            canvas = Image.new('RGB', (1024, 640), '#182330')
            draw = ImageDraw.Draw(canvas)
            draw.text((15, 4), '同一场景：酒瓶 → 碗｜此次未新增训练', font=font, fill='white')
            for side, title in enumerate(('纯 π0.5：目标错误（架子）', 'HAMSTER＋π0.5＋RGB-D/OSC')):
                draw.text((side*512+12, 39), title, font=small, fill='#ffdf89')
                if index < counts[side]:
                    last[side] = readers[side].get_data(index)
                assert last[side].shape == (512, 512, 3)
                canvas.paste(Image.fromarray(last[side]), (side*512, 72))
                label = (f'控制步 {index}/{counts[side]-1}' if index < counts[side] else
                         f'已结束于 {counts[side]-1} 步；保留尾帧')
                draw.text((side*512+12, 587), label, font=small, fill='white')
            draw.text((12, 614), '按控制步同步，20 fps；不是墙钟时间／推理延迟对比', font=small, fill='#c6d5e3')
            writer.append_data(np.asarray(canvas))
            if index == max(counts)-1:
                canvas.save(args.output / 'comparison-final.png')
    for reader in readers:
        reader.close()
    receipt = dict(fps=20, source_frames=counts, output_frames=max(counts),
                   alignment='same initial state, control index; hold last frame after trial ends',
                   training=False, input_image_sha256=reports[0]['image_sha256'],
                   sources=[dict(directory=str(path), video_sha256=hashlib.sha256(
                       (path / 'rollout.mp4').read_bytes()).hexdigest()) for path in paths],
                   output_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    (args.output / 'manifest.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
