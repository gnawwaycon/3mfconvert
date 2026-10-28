# Spinner art → 3MF

`spinner_art.py` takes a flat-color image (e.g. from Gemini) and a blank part, and writes a
Bambu Studio 3MF with the art as a multi-color layer on the part. It replaces the CorelDRAW
trace and the per-color extrudes in Onshape.

## Setup (once)

```bash
uv venv --python 3.12 .venv && VIRTUAL_ENV=.venv uv pip install -r requirements.txt
```

## Use

```bash
.venv/bin/python spinner_art.py rippon.jpg "spinner template Magnetic - Part 1.3mf" -o output/ripon.3mf
```

It prints the colors it found and which filament slot each one uses, and writes:

- `output/ripon.3mf`: open it in Bambu Studio. It's one object: the base part plus one part
  per color, each already assigned to a filament slot.
- `output/ripon.preview.png`: the art layer as it will look on the finished part.

### Templates

Export the blank part from Onshape as 3MF (or STL). The art goes on its lowest flat face by
default, printed face-down on the plate:

- **Template with a gap under it** (like `spinner template Magnetic`, whose body starts at
  z = 1 mm): the art fills that gap exactly.
- **Template sitting on z = 0**: a 1 mm layer (`--layer`) is added below it.
- `--face top` puts the art on the highest flat face instead (e.g. the bearing cap STL).

The art is centered on the biggest circle that fits the face, sized to leave `--margin` mm
(default 3) around it, and cut around any holes. On a bottom face it's mirrored so it reads
correctly once the print is flipped over.

### Useful options

| Option | What it does |
|---|---|
| `--margin 1` | Smaller border; use this on small caps |
| `--size 60` | Art diameter in mm (overrides `--margin`) |
| `--rotate 90`, `--offset X Y` | Turn or move the art |
| `--colors 4` | Force the number of colors, background included (default: auto) |
| `--filaments "#FFFFFF,#000000,#C12E1F"` | Your AMS slots in order. Each image color goes to the closest slot, so similar shades merge |
| `--base-slot 5` | Slot for the part itself and the image background, e.g. a colored backing instead of white (use with `--filaments`) |
| `--layer 0.6` | Art layer thickness |

### Bearing cap example

The cap's peg came out slightly too tight in the bearing, so `resize_peg.py` made an 8.0 mm
version (was 8.2 mm) of the blank cap:

```bash
.venv/bin/python resize_peg.py blank_spinner_bearing_wing_cap.stl --diameter 8.0
```

Then the art goes on the cap's top face, using the 5 loaded filaments in AMS order
(light pink, dark pink, gray, purple, white):

```bash
.venv/bin/python spinner_art.py art/tinkaton.png blank_spinner_bearing_wing_cap_peg8mm.stl --face top --margin 1 --filaments "#F4B6C8,#D63A7A,#8C8F94,#5A2D82,#FFFFFF" -o output/tinkaton_cap.3mf
```

`art/tinkaton.svg` is hand-drawn for the 21 mm cap: lines at least 0.5 mm wide, only the
loaded colors. Re-render it to `art/tinkaton.png` after editing:

```bash
qlmanage -t -s 1024 -o art art/tinkaton.svg && mv art/tinkaton.svg.png art/tinkaton.png
```

### Getting clean results

- Ask Gemini for *"flat vector style, N solid colors, no gradients or shading, bold outlines,
  plain white background"*. The white background becomes the base color.
- Watch for the **"thinner than the 0.4 mm nozzle"** note in the printout. Lines thinner
  than the nozzle won't print. On the 21 mm bearing cap, Gemini outlines come out at about
  0.2 mm, so use bolder art or a bigger face.
- White base + 4 colors needs 5 filaments. Use `--filaments` with your 4 AMS colors to merge
  similar shades down.

## How the Ripon test compared

The script rebuilt `Ripon spinner(2).3mf` from `rippon.jpg` and the blank template:

| | Hand-made | Script | Overlap |
|---|---|---|---|
| Black | 1328 mm² | 1316 mm² | 92% |
| Red | 945 mm² | 945 mm² | 98% |
| White fill | 3666 mm² | 3686 mm² | 97% |

These numbers are from a run at the same art size as the hand-made file (`--size 78.76`).
Edges are on average 63 µm apart, smaller than one pixel of the image and well under the nozzle
width. The base body is identical to the template, and the color parts don't overlap. Bambu
Studio's command-line slicer (X1C, 0.16 mm) sliced the file with each color on the right
filament: art in the first 5 layers, only the base filament above. See
`output/ripon_comparison.png`.
