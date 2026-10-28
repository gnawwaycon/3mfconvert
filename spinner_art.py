#!/usr/bin/env python3
"""Put flat-color image art onto a blank 3D-print template and write a Bambu Studio 3MF.

    python spinner_art.py rippon.jpg "spinner template Magnetic - Part 1.3mf"

Steps: split the image into solid colors, trace each color into outlines, scale and
center them on a flat face of the template, extrude each color into the art layer, and
write one object whose parts are already assigned to filament slots. Replaces the
CorelDRAW trace + Onshape per-color extrude steps.
"""
import argparse
import math
import sys
import uuid
import zipfile
from datetime import date
from pathlib import Path

import cv2
import manifold3d as mf
import numpy as np
import shapely
from lxml import etree
from shapely import affinity
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.polygon import orient
from sklearn.cluster import KMeans

NS_CORE = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
NS_MAT = "http://schemas.microsoft.com/3dmanufacturing/material/2015/02"
NS_PROD = "http://schemas.microsoft.com/3dmanufacturing/production/2015/06"
UNIT_MM = {"micron": 0.001, "millimeter": 1.0, "centimeter": 10.0, "inch": 25.4, "foot": 304.8, "meter": 1000.0}
BAMBU_VERSION = "02.08.02.60"


# ---------------------------------------------------------------- template ---

def _transform(attr):
    """3MF transform string -> 4x4 matrix acting on column vectors."""
    if not attr:
        return np.eye(4)
    m = np.array([float(x) for x in attr.split()]).reshape(4, 3)
    out = np.eye(4)
    out[:3, :3] = m[:3].T
    out[:3, 3] = m[3]
    return out


def load_template(path):
    """Return (vertices_mm, faces, color_hex) with every body in the file merged."""
    path = Path(path)
    if path.suffix.lower() != ".3mf":
        import trimesh
        mesh = trimesh.load(path, force="mesh")
        return np.asarray(mesh.vertices, float), np.asarray(mesh.faces, np.int64), None

    zf = zipfile.ZipFile(path)
    roots = {"/" + n: etree.fromstring(zf.read(n)) for n in zf.namelist() if n.endswith(".model")}
    main = "/3D/3dmodel.model"
    objects, colors = {}, {}
    for part, root in roots.items():
        scale = UNIT_MM.get(root.get("unit", "millimeter"), 1.0)
        for g in root.iter(f"{{{NS_MAT}}}colorgroup"):
            colors[(part, g.get("id"))] = [c.get("color") for c in g.iter(f"{{{NS_MAT}}}color")]
        for g in root.iter(f"{{{NS_CORE}}}basematerials"):
            colors[(part, g.get("id"))] = [b.get("displaycolor") for b in g.iter(f"{{{NS_CORE}}}base")]
        for o in root.iter(f"{{{NS_CORE}}}object"):
            objects[(part, o.get("id"))] = (o, scale)

    verts, faces, color = [], [], None

    def walk(key, tf):
        nonlocal color
        obj, scale = objects[key]
        mesh = obj.find(f"{{{NS_CORE}}}mesh")
        if mesh is not None:
            v = np.array([[float(e.get(a)) for a in "xyz"] for e in mesh.iter(f"{{{NS_CORE}}}vertex")]) * scale
            f = np.array([[int(e.get(a)) for a in ("v1", "v2", "v3")] for e in mesh.iter(f"{{{NS_CORE}}}triangle")])
            v = (np.c_[v, np.ones(len(v))] @ tf.T)[:, :3]
            faces.append(f + sum(len(x) for x in verts))
            verts.append(v)
            if color is None and obj.get("pid"):
                group = colors.get((key[0], obj.get("pid")))
                if group:
                    color = group[int(obj.get("pindex", 0))][:7]
        for c in obj.iter(f"{{{NS_CORE}}}component"):
            child = (c.get(f"{{{NS_PROD}}}path") or key[0], c.get("objectid"))
            t = _transform(c.get("transform"))
            t[:3, 3] *= scale
            walk(child, tf @ t)

    main_scale = UNIT_MM.get(roots[main].get("unit", "millimeter"), 1.0)
    for item in roots[main].iter(f"{{{NS_CORE}}}item"):
        t = _transform(item.get("transform"))
        t[:3, 3] *= main_scale
        walk((main, item.get("objectid")), t)
    return np.vstack(verts), np.vstack(faces), color


def to_manifold(v, f):
    mesh = mf.Mesh(vert_properties=np.ascontiguousarray(v, np.float32),
                   tri_verts=np.ascontiguousarray(f, np.uint32))
    mesh.merge()
    return mf.Manifold(mesh)


def find_art_face(v, f, side):
    """Outline (shapely, mm) and height of the lowest/highest flat face of the template."""
    tri = v[f]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nz = n[:, 2] / (np.linalg.norm(n, axis=1) + 1e-12)
    if side == "bottom":
        z0 = v[:, 2].min()
        sel = (nz < -0.999) & np.all(np.abs(tri[:, :, 2] - z0) < 1e-3, axis=1)
    else:
        z0 = v[:, 2].max()
        sel = (nz > 0.999) & np.all(np.abs(tri[:, :, 2] - z0) < 1e-3, axis=1)
    rings = tri[sel][:, :, :2]
    rings = np.concatenate([rings, rings[:, :1]], axis=1)
    face = shapely.union_all(shapely.polygons(rings), grid_size=1e-4)
    # CAD meshes leave hairline cracks between triangles; close them, keep real holes.
    face = face.buffer(0.01, join_style="mitre").buffer(-0.01, join_style="mitre")
    face = shapely.union_all([Polygon(p.exterior, [r for r in p.interiors if Polygon(r).area > 0.05])
                              for p in getattr(face, "geoms", [face])])
    return z0, face


def face_center(face):
    """Center and radius of the biggest circle that fits the face (holes ignored)."""
    biggest = max(getattr(face, "geoms", [face]), key=lambda p: p.area)
    mic = shapely.maximum_inscribed_circle(Polygon(biggest.exterior), tolerance=0.01)
    (cx, cy), _ = mic.coords
    return (cx, cy), mic.length


# ------------------------------------------------------------------- image ---

def load_image(path):
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        sys.exit(f"Could not read image: {path}")
    alpha = None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        alpha = img[:, :, 3] > 127
        img = img[:, :, :3]
        if alpha.all():  # an alpha channel with nothing transparent: use the border color as background
            alpha = None
    return img, alpha


def lab_to_true(lab):
    """OpenCV 8-bit Lab -> CIE Lab, so distances are delta-E."""
    lab = np.asarray(lab, float)
    return np.c_[lab[..., 0] * 100 / 255, lab[..., 1] - 128, lab[..., 2] - 128]


def hex_to_lab(h):
    h = h.lstrip("#")
    bgr = np.uint8([[[int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)]]])
    return lab_to_true(cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).reshape(1, 3))[0]


def lab_to_hex(lab8):
    rgb = cv2.cvtColor(np.uint8([[lab8]]), cv2.COLOR_LAB2RGB)[0, 0]
    return "#" + "".join(f"{c:02X}" for c in rgb)


def nearest(lab, centers):
    d = ((lab[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
    return d.argmin(1)


def pick_palette(lab, sample, shape, n_colors):
    """Cluster centers (OpenCV Lab). With n_colors=None, find the real flat colors and
    drop the in-between shades that anti-aliasing creates along edges."""
    if n_colors:
        return KMeans(n_colors, n_init=4, random_state=0).fit(sample).cluster_centers_
    centers = KMeans(10, n_init=2, random_state=0).fit(sample).cluster_centers_
    labels = nearest(lab, centers).reshape(shape)
    kernel = np.ones((7, 7), np.uint8)
    core = np.array([cv2.erode((labels == k).astype(np.uint8), kernel).sum() for k in range(len(centers))])
    keep = [k for k in np.argsort(-core) if core[k] >= max(300, 2e-4 * labels.size)]
    merged = []
    for k in keep:  # most solid colors first; fold near-duplicates into them
        if all(np.linalg.norm(lab_to_true(centers[k]) - lab_to_true(centers[m])) >= 10 for m in merged):
            merged.append(k)
    return KMeans(len(merged), init=centers[merged], n_init=1).fit(sample).cluster_centers_


def majority_filter(labels, n, size=5):
    votes = np.stack([cv2.blur((labels == k).astype(np.float32), (size, size)) for k in range(n)])
    return votes.argmax(0)


def remove_specks(labels, min_px):
    """Give every blob smaller than min_px to the color that surrounds it."""
    out = labels.copy()
    for k in np.unique(labels):
        count, comp, stats, _ = cv2.connectedComponentsWithStats((out == k).astype(np.uint8), connectivity=4)
        for i in np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] < min_px) + 1:
            x, y, w, h = stats[i, :4]
            x0, y0, x1, y1 = max(x - 2, 0), max(y - 2, 0), x + w + 2, y + h + 2
            blob = comp[y0:y1, x0:x1] == i
            ring = cv2.dilate(blob.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~blob
            around = out[y0:y1, x0:x1][ring]
            if around.size:
                out[y0:y1, x0:x1][blob] = np.bincount(around).argmax()
    return out


def separate_colors(img, alpha, n_colors):
    """Return (label image, palette as OpenCV Lab, background label or -1)."""
    smooth = cv2.bilateralFilter(img, 9, 40, 9)
    lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    valid = alpha.reshape(-1) if alpha is not None else np.ones(len(lab), bool)
    pool = np.flatnonzero(valid)
    sample = lab[np.random.default_rng(0).choice(pool, min(60000, len(pool)), replace=False)]
    centers = pick_palette(lab, sample, img.shape[:2], n_colors)
    labels = majority_filter(nearest(lab, centers).reshape(img.shape[:2]), len(centers))
    if alpha is not None:
        labels[~alpha] = len(centers)  # transparent pixels become the base
        return labels, centers, len(centers)
    border = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    return labels, centers, int(np.bincount(border).argmax())


def trace(mask, up=4, sigma=0.8):
    """Smooth outlines of a binary mask, in image pixel coordinates."""
    m = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma)
    m = cv2.resize(m, None, fx=up, fy=up, interpolation=cv2.INTER_LINEAR)
    contours, hier = cv2.findContours((m >= 0.5).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for i, c in enumerate(contours):
        if hier[0][i][3] != -1 or len(c) < 3:
            continue
        holes, j = [], hier[0][i][2]
        while j != -1:
            if len(contours[j]) >= 3:
                holes.append((contours[j][:, 0, :] + 0.5) / up - 0.5)
            j = hier[0][j][0]
        polys.append(Polygon((c[:, 0, :] + 0.5) / up - 0.5, holes).buffer(0))
    return shapely.union_all(polys).simplify(0.25)


# ------------------------------------------------------------------ output ---

def extrude(geom, z0, height):
    contours = []
    for p in getattr(geom, "geoms", [geom]):
        if p.is_empty or p.geom_type != "Polygon":
            continue
        p = orient(p, 1.0)
        contours.append(np.asarray(p.exterior.coords)[:-1])
        contours += [np.asarray(r.coords)[:-1] for r in p.interiors]
    if not contours:
        return mf.Manifold()
    return mf.Manifold.extrude(mf.CrossSection(contours, mf.FillRule.EvenOdd), height).translate((0, 0, z0))


def write_3mf(path, name, parts, bed_center):
    """parts: [dict(name, slot, color, manifold)] -> Bambu Studio project-style 3MF."""
    meshes = []
    for p in parts:
        m = p["manifold"].to_mesh()
        meshes.append((np.asarray(m.vert_properties)[:, :3].astype(float), np.asarray(m.tri_verts)))
    allv = np.vstack([v for v, _ in meshes])
    lo, hi = allv.min(0), allv.max(0)
    shift = np.array([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])

    obj_xml = []
    for i, (v, f) in enumerate(meshes, start=1):
        v = v + shift
        vx = "".join(f'<vertex x="{x:.5f}" y="{y:.5f}" z="{z:.5f}"/>' for x, y, z in v)
        tx = "".join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in f)
        obj_xml.append(f'<object id="{i}" p:UUID="{uuid.uuid4()}" type="model"><mesh>'
                       f"<vertices>{vx}</vertices><triangles>{tx}</triangles></mesh></object>")
    head = ('<?xml version="1.0" encoding="UTF-8"?>\n<model unit="millimeter" xml:lang="en-US" '
            f'xmlns="{NS_CORE}" xmlns:BambuStudio="http://schemas.bambulab.com/package/2021" '
            f'xmlns:p="{NS_PROD}" requiredextensions="p">\n')
    objects_model = head + ' <metadata name="BambuStudio:3mfVersion">1</metadata>\n <resources>\n' \
        + "\n".join(obj_xml) + "\n </resources>\n <build/>\n</model>\n"

    oid = len(parts) + 1
    comps = "".join(f'<component p:path="/3D/Objects/object_1.model" objectid="{i}" p:UUID="{uuid.uuid4()}" '
                    f'transform="1 0 0 0 1 0 0 0 1 0 0 0"/>' for i in range(1, oid))
    today = date.today().isoformat()
    main_model = head + (
        f' <metadata name="Application">BambuStudio-{BAMBU_VERSION}</metadata>\n'
        ' <metadata name="BambuStudio:3mfVersion">1</metadata>\n'
        f' <metadata name="CreationDate">{today}</metadata>\n <metadata name="ModificationDate">{today}</metadata>\n'
        f' <metadata name="Title">{name}</metadata>\n'
        f' <resources>\n  <object id="{oid}" p:UUID="{uuid.uuid4()}" type="model"><components>{comps}</components></object>\n </resources>\n'
        f' <build p:UUID="{uuid.uuid4()}">\n  <item objectid="{oid}" p:UUID="{uuid.uuid4()}" '
        f'transform="1 0 0 0 1 0 0 0 1 {bed_center[0]} {bed_center[1]} 0" printable="1"/>\n </build>\n</model>\n')

    part_xml = "".join(
        f'    <part id="{i}" subtype="normal_part">\n'
        f'      <metadata key="name" value="{p["name"]}"/>\n'
        '      <metadata key="matrix" value="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"/>\n'
        f'      <metadata key="extruder" value="{p["slot"]}"/>\n    </part>\n'
        for i, p in enumerate(parts, start=1))
    settings = ('<?xml version="1.0" encoding="UTF-8"?>\n<config>\n'
                f'  <object id="{oid}">\n    <metadata key="name" value="{name}"/>\n'
                f'    <metadata key="extruder" value="{parts[0]["slot"]}"/>\n{part_xml}  </object>\n'
                '  <plate>\n    <metadata key="plater_id" value="1"/>\n    <metadata key="plater_name" value=""/>\n'
                '    <metadata key="locked" value="false"/>\n    <model_instance>\n'
                f'      <metadata key="object_id" value="{oid}"/>\n      <metadata key="instance_id" value="0"/>\n'
                '    </model_instance>\n  </plate>\n</config>\n')

    rel = '<Relationship Target="{}" Id="rel-1" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>'
    rels = ('<?xml version="1.0" encoding="UTF-8"?>\n<Relationships '
            'xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n {}\n</Relationships>\n')
    types = ('<?xml version="1.0" encoding="UTF-8"?>\n<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
             ' <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>\n'
             ' <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>\n'
             ' <Default Extension="png" ContentType="image/png"/>\n</Types>\n')
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", types)
        z.writestr("_rels/.rels", rels.format(rel.format("/3D/3dmodel.model")))
        z.writestr("3D/3dmodel.model", main_model)
        z.writestr("3D/_rels/3dmodel.model.rels", rels.format(rel.format("/3D/Objects/object_1.model")))
        z.writestr("3D/Objects/object_1.model", objects_model)
        z.writestr("Metadata/model_settings.config", settings)


def save_preview(path, face, regions, base_color, side):
    """Picture of the art layer as it will look on the finished part."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MPath

    def patch(geom, color, **kw):
        for p in getattr(geom, "geoms", [geom]):
            if p.is_empty or p.geom_type != "Polygon":
                continue
            rings = [np.asarray(p.exterior.coords)] + [np.asarray(r.coords) for r in p.interiors]
            codes = np.concatenate([[MPath.MOVETO] + [MPath.LINETO] * (len(r) - 2) + [MPath.CLOSEPOLY] for r in rings])
            if side == "bottom":  # show it the way you'll see it after flipping the print over
                rings = [r * [1, -1] for r in rings]
            ax.add_patch(PathPatch(MPath(np.concatenate(rings), codes), facecolor=color, **kw))

    fig, ax = plt.subplots(figsize=(7, 7))
    patch(face, base_color, edgecolor="#555", lw=0.6)
    for r in regions:
        patch(r["geom"], r["color"], edgecolor="none")
    ax.autoscale()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Art layer as seen on the finished part", fontsize=10)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#9fb4c8")
    plt.close(fig)


# -------------------------------------------------------------------- main ---

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", help="art image (PNG/JPG; transparent PNG backgrounds become the base color)")
    ap.add_argument("template", help="blank part: 3MF exported from Onshape, or STL")
    ap.add_argument("-o", "--output", help="output 3MF (default: <image name>.3mf next to the image)")
    ap.add_argument("--face", choices=["bottom", "top"], default="bottom",
                    help="flat face to put the art on (default bottom: printed face-down on the plate)")
    ap.add_argument("--layer", type=float,
                    help="art layer thickness in mm (default: fill the gap under a bottom face, else 1.0)")
    ap.add_argument("--margin", type=float, default=3.0, help="space between art and face edge, mm (default 3)")
    ap.add_argument("--size", type=float, help="art diameter in mm (overrides --margin)")
    ap.add_argument("--rotate", type=float, default=0.0, help="rotate the art, degrees counter-clockwise")
    ap.add_argument("--offset", type=float, nargs=2, default=(0, 0), metavar=("X", "Y"),
                    help="shift the art from the face center, mm")
    ap.add_argument("--colors", type=int, help="number of colors in the image, background included (default: auto)")
    ap.add_argument("--filaments", help="comma-separated AMS slot colors, e.g. '#FFFFFF,#000000,#C12E1F'; "
                                        "each image color is matched to the closest slot")
    ap.add_argument("--base-slot", type=int, help="slot for the part and the image background "
                                                  "(default: the slot closest to the background color)")
    ap.add_argument("--min-feature", type=float, default=0.15, help="drop specks smaller than this many mm² (default 0.15)")
    ap.add_argument("--nozzle", type=float, default=0.4, help="nozzle width used for the thin-line warning (default 0.4)")
    ap.add_argument("--bed-center", type=float, nargs=2, default=(128, 128), metavar=("X", "Y"))
    args = ap.parse_args()

    image_path, template_path = Path(args.image), Path(args.template)
    out = Path(args.output) if args.output else image_path.with_suffix(".3mf")
    if out.suffix.lower() != ".3mf":
        out = out.with_name(out.name + ".3mf")

    # Template and the face the art goes on
    v, f, template_color = load_template(template_path)
    z0, face = find_art_face(v, f, args.face)
    if face.is_empty:
        sys.exit(f"No flat {args.face} face found on {template_path}")
    (cx, cy), radius = face_center(face)
    gap = z0 if args.face == "bottom" else 0.0
    layer = args.layer or (gap if gap > 0.05 else 1.0)
    diameter = args.size or 2 * (radius - args.margin)
    print(f"Template: {args.face} face at z={z0:.2f} mm, fits a {2 * radius:.1f} mm circle at ({cx:.2f}, {cy:.2f})")
    print(f"Art: {diameter:.1f} mm across, {layer:.2f} mm thick")

    # Image -> solid colors
    img, alpha = load_image(image_path)
    labels, centers, bg = separate_colors(img, alpha, args.colors)
    labels = remove_specks(labels, 20)
    content = (labels != bg).astype(np.uint8)
    pts, _ = cv2.findContours(content, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not pts:
        sys.exit("The image has no art apart from the background.")
    (icx, icy), irad = cv2.minEnclosingCircle(np.vstack(pts))
    mm_per_px = diameter / 2 / irad
    labels = remove_specks(labels, max(20, args.min_feature / mm_per_px ** 2))

    # Image colors -> filament slots. The background always goes to the base body.
    color_hex = {k: lab_to_hex(c) for k, c in enumerate(centers.astype(np.uint8))}
    if args.filaments:
        slots = [s.strip() if s.strip().startswith("#") else "#" + s.strip() for s in args.filaments.split(",")]
        slot_lab = np.array([hex_to_lab(s) for s in slots])
        pick = lambda k: int(np.argmin(((slot_lab - lab_to_true(centers[k])) ** 2).sum(1))) + 1
        base_slot = args.base_slot or (pick(bg) if bg < len(centers) else 1)
        slot_of = {k: pick(k) for k in range(len(centers)) if k != bg}
        slot_color = {i + 1: s for i, s in enumerate(slots)}
    else:
        base_slot = 1
        order = sorted((k for k in range(len(centers)) if k != bg), key=lambda k: -(labels == k).sum())
        slot_of = {k: i + 2 for i, k in enumerate(order)}
        slot_color = {base_slot: color_hex.get(bg, template_color or "#EAEAEA")}
        slot_color.update({slot_of[k]: color_hex[k] for k in order})

    # Trace each color and place it on the face. On a bottom face the art is mirrored
    # so it reads correctly once the print is flipped over.
    theta = math.radians(args.rotate)
    a, b = mm_per_px * math.cos(theta), mm_per_px * math.sin(theta)
    flip = -1 if args.face == "bottom" else 1
    # viewer coords: u = s(px-icx), w = -s(py-icy), rotated; model y = flip * w
    tx, ty = cx + args.offset[0], cy + flip * args.offset[1]
    matrix = [a, b, flip * b, -flip * a,
              tx - a * icx - b * icy, ty - flip * (b * icx - a * icy)]

    regions = {}
    for slot in sorted(set(slot_of.values()) - {base_slot}):
        mask = np.isin(labels, [k for k, s in slot_of.items() if s == slot])
        regions[slot] = affinity.affine_transform(trace(mask), matrix)
    # Neighbors overlap slightly so no gaps open between colors; later (smaller) colors win.
    overlap = 0.3 * mm_per_px
    order = sorted(regions, key=lambda s: -regions[s].area)
    for i, s in enumerate(order):
        later = shapely.union_all([regions[t].buffer(overlap) for t in order[i + 1:]]) if i + 1 < len(order) else None
        g = regions[s].buffer(overlap)
        if later is not None:
            g = g.difference(later)
        g = g.intersection(face)
        regions[s] = shapely.union_all([p for p in getattr(g, "geoms", [g])
                                        if p.geom_type == "Polygon" and p.area >= args.min_feature])
    base_layer = face.difference(shapely.union_all(list(regions.values())))

    # Solids: template + base-color part of the art layer form one body, plus one per color.
    z_lo = z0 - layer if args.face == "bottom" else z0
    base = to_manifold(v, f)
    if base.status() != mf.Error.NoError:
        sys.exit(f"Template mesh is not watertight ({base.status()}); re-export it from Onshape.")
    base = base + extrude(base_layer, z_lo, layer)
    parts = [dict(name="base", slot=base_slot, color=slot_color[base_slot], manifold=base)]
    print(f"\n{'slot':>4}  {'color':8}  {'area mm²':>8}  note")
    print(f"{base_slot:>4}  {slot_color[base_slot]:8}  {base_layer.area:8.1f}  base body + background")
    for s in sorted(regions):
        g = regions[s]
        if g.is_empty:
            continue
        thin = g.difference(g.buffer(-args.nozzle / 2).buffer(args.nozzle / 2)).area / g.area
        note = f"{thin:.0%} of it is thinner than the {args.nozzle} mm nozzle" if thin > 0.05 else ""
        print(f"{s:>4}  {slot_color[s]:8}  {g.area:8.1f}  {note}")
        parts.append(dict(name=f"color {s} {slot_color[s]}", slot=s, color=slot_color[s],
                          manifold=extrude(g, z_lo, layer)))

    write_3mf(out, image_path.stem, parts, args.bed_center)
    preview = out.with_suffix(".preview.png")
    save_preview(preview, face, [dict(geom=regions[s], color=slot_color[s]) for s in regions],
                 slot_color[base_slot], args.face)
    print(f"\nWrote {out}\nPreview {preview}")


if __name__ == "__main__":
    main()
