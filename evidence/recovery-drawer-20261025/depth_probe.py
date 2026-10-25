"""Offline RGB-D inspection of recorded simulation states; never controls motion."""
import json
import sys
from pathlib import Path
import numpy as np
from probe import scene, ROOT
sys.path.insert(0, str(ROOT))
from vla.visual_control import VisualControl
from vla.semantic_grasp import image_bytes

source = Path(sys.argv[1])
env, obs = scene()
try:
    for number in (1, 3):
        metadata = json.loads((source / f'step-{number:02d}/result.json').read_text())
        geometry = metadata['geometry']
        handle = np.array(geometry['handle'])
        normal = np.array(geometry['outward_normal'])
        tangent = np.cross(normal, [0., 0., 1.])
        for state in (0, 1, 2, 3):
            if state:
                obs = env.set_init_state(np.load(source / f'state-{state:02d}.npy'))
            else:
                env.close()
                env, obs = scene()
            skill = VisualControl(env, obs, None, source / f'depth-{number}-{state}')
            rgb, cloud = skill.capture()
            np.savez_compressed(skill.output / 'cloud.npz', cloud=cloud)
            (skill.output / 'image.png').write_bytes(image_bytes(rgb))
            points = cloud.reshape(-1, 3)
            delta = points - handle
            # Drawer front below its horizontal handle, across its width.
            selected = points[(abs(delta @ tangent) < .075) & (delta[:, 2] > -.04) & (delta[:, 2] < -.012)]
            coordinates = (selected - handle) @ normal
            hist, edges = np.histogram(coordinates, bins=np.arange(-.25, .251, .005))
            ids = np.argsort(hist)[-5:][::-1]
            print(number, state, len(selected), [(round(float(edges[i]), 3), int(hist[i])) for i in ids], flush=True)
finally:
    env.close()
