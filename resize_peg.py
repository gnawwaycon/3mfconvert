#!/usr/bin/env python3
"""Change the diameter of the round peg at the bottom of a part (e.g. the cap's bearing peg).

    python resize_peg.py blank_spinner_bearing_wing_cap.stl --diameter 8.0

Finds the cylinder that touches z = min, and moves its wall in or out to the new diameter.
Writes <name>_peg<diameter>.stl next to the input unless -o is given.
"""
import argparse
from pathlib import Path

import numpy as np
import trimesh


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("part", help="STL of the part")
    ap.add_argument("--diameter", type=float, required=True, help="new peg diameter in mm")
    ap.add_argument("-o", "--output")
    args = ap.parse_args()

    mesh = trimesh.load(args.part, force="mesh")
    v = mesh.vertices.copy()
    bottom = np.abs(v[:, 2] - v[:, 2].min()) < 1e-3
    center = (v[bottom, :2].min(0) + v[bottom, :2].max(0)) / 2
    r = np.linalg.norm(v[:, :2] - center, axis=1)
    old = r[bottom].max()
    wall = np.abs(r - old) < 0.01
    # the peg wall is every vertex on that radius from the bottom up to where the radius changes
    top = v[wall, 2].max()
    if np.any(bottom & (r < old - 0.01) & (r > 1e-6)):
        raise SystemExit("The bottom face isn't a plain disc; this script only handles a simple round peg.")
    v[wall] = np.c_[center + (v[wall, :2] - center) * (args.diameter / 2 / old), v[wall, 2]]
    new = trimesh.Trimesh(v, mesh.faces, process=False)
    if not new.is_watertight or new.volume <= 0:
        raise SystemExit("Resizing broke the mesh; change the peg in Onshape instead.")

    out = Path(args.output) if args.output else Path(args.part).with_name(
        f"{Path(args.part).stem}_peg{args.diameter:g}mm.stl")
    new.export(out)
    print(f"Peg: {2 * old:.2f} mm -> {args.diameter:.2f} mm across, {top - v[:, 2].min():.2f} mm tall")
    print(f"Volume {mesh.volume:.1f} -> {new.volume:.1f} mm³\nWrote {out}")


if __name__ == "__main__":
    main()
