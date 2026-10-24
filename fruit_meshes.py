"""Deterministic visual fruit meshes in metres; collision stays in scene.py."""
import xml.etree.ElementTree as ET

import numpy as np


def add_fruit_meshes(asset):
    """Shared ring vertices give smooth lighting without third-party assets."""
    sides, rings = 64, 49
    theta = np.arange(sides) * 2 * np.pi / sides
    for name in ("apple_skin", "banana_skin"):
        vertices = []
        for i in range(rings):
            t = (i + .001) / (rings - 1 + .002)
            if name == "apple_skin":
                phi = np.pi * t
                # Broad shoulders, five subtle lobes, indented stem and blossom.
                radius = .027 * np.sin(phi) * (1 + .055 * np.cos(phi))
                radial = radius * (1 + .035 * np.cos(5 * theta) * np.sin(phi))
                z = .026 * np.cos(phi)
                z -= .006 * np.exp(-(phi / .23) ** 2)
                z += .003 * np.exp(-((np.pi - phi) / .23) ** 2)
                points = np.column_stack((radial * np.cos(theta), radial * np.sin(theta),
                                          np.full(sides, z)))
            else:
                x = .12 * (t - .5)
                y = .022 * (2 * t - 1) ** 2
                slope = .088 * (2 * t - 1) / .12
                normal = np.array([-slope, 1, 0]) / np.sqrt(1 + slope ** 2)
                # A continuous bent peel with tapered ends and five soft ridges.
                radius = .0205 * np.sin(np.pi * t) ** .55
                radial = radius * (1 + .055 * np.cos(5 * theta))
                points = (np.array([x, y, 0]) +
                          radial[:, None] * np.cos(theta)[:, None] * normal +
                          (radial * np.sin(theta))[:, None] * np.array([0, 0, 1]))
            vertices.extend(points)
        faces = []
        for i in range(rings - 1):
            for j in range(sides):
                a, b = i * sides + j, i * sides + (j + 1) % sides
                c, d = a + sides, b + sides
                faces.extend(((a, c, b), (b, c, d)))
        # End caps are hidden under the tiny stem/blossom details.
        for j in range(1, sides - 1):
            faces.extend(((0, j, j + 1),
                          ((rings - 1) * sides, (rings - 1) * sides + j + 1,
                           (rings - 1) * sides + j)))
        if name == "banana_skin":
            faces = [(a, c, b) for a, b, c in faces]
        ET.SubElement(asset, "mesh", name=name,
                      vertex=" ".join(f"{v:.7f}" for point in vertices for v in point),
                      face=" ".join(str(v) for face in faces for v in face))
