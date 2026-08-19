"""Render the compact paper subfigure for the v5 earring recovery module."""

from math import atan2, cos, pi, sin
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets" / "v5_ear_detail_restoration.png"
SOURCE = ROOT / "docs" / "assets" / "v5_ear_source_example.png"
S = 2
GREEN, DARK, MUTED, BLUE = "#147f5e", "#26353c", "#52636c", "#7a9fc4"


def f(size, bold=False):
    return ImageFont.truetype("arialbd.ttf" if bold else "arial.ttf", size * S)


def p(xy):
    return tuple(round(v * S) for v in xy)


def txt(d, xy, value, size=11, fill=DARK, bold=False, anchor=None):
    d.text(p(xy), value, font=f(size, bold), fill=fill, anchor=anchor)


def box(d, xy, fill, outline=None, width=1, radius=0):
    d.rounded_rectangle(p(xy), radius=radius * S, fill=fill, outline=outline, width=width * S)


def line(d, points, fill=GREEN, width=2, dash=None):
    points = [p(pt) for pt in points]
    if not dash:
        d.line(points, fill=fill, width=width * S, joint="curve")
        return
    for a, b in zip(points, points[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = (dx * dx + dy * dy) ** 0.5
        if length == 0:
            continue
        ux, uy, cur, i, draw = dx / length, dy / length, 0.0, 0, True
        while cur < length:
            nxt = min(length, cur + dash[i % len(dash)] * S)
            if draw:
                d.line(((round(a[0] + ux * cur), round(a[1] + uy * cur)),
                        (round(a[0] + ux * nxt), round(a[1] + uy * nxt))), fill=fill, width=width * S)
            cur, i, draw = nxt, i + 1, not draw


def arrow(d, points, fill=GREEN, width=2, dash=None):
    line(d, points, fill, width, dash)
    (x1, y1), (x2, y2) = points[-2], points[-1]
    angle, tip, h = atan2(y2 - y1, x2 - x1), p((x2, y2)), 7 * S
    d.polygon([tip,
               (round(tip[0] - h * cos(angle - pi / 6)), round(tip[1] - h * sin(angle - pi / 6))),
               (round(tip[0] - h * cos(angle + pi / 6)), round(tip[1] - h * sin(angle + pi / 6)))], fill=fill)


def photo(canvas, image, xy, opacity=1):
    x0, y0, x1, y1 = p(xy)
    crop = ImageOps.fit(image, (x1 - x0, y1 - y0), Image.Resampling.LANCZOS, centering=(0.5, 0.42))
    if opacity < 1:
        crop = Image.blend(Image.new("RGB", crop.size, "#fff"), crop, opacity)
    canvas.paste(crop, (x0, y0))


def circle(d, c, radius, fill="#fff", outline=GREEN, width=1):
    x, y = p(c)
    d.ellipse((x - radius * S, y - radius * S, x + radius * S, y + radius * S), fill=fill, outline=outline, width=width * S)


def plates(d, x, y, count=2, blue=False):
    colors = ("#a9c9ec", "#7397c2", "#dceafa") if blue else ("#a1d7b4", "#278664", "#d8f0e1")
    for i in range(count):
        xx, yy, height = x + 23 * i, y + 8 * i, 56 - 9 * i
        d.polygon([p(pt) for pt in [(xx, yy + 12), (xx + 12, yy), (xx + 12, yy + height), (xx, yy + height + 12)]], fill=colors[0], outline="#477766")
        d.polygon([p(pt) for pt in [(xx + 12, yy), (xx + 19, yy + 5), (xx + 19, yy + height + 5), (xx + 12, yy + height)]], fill=colors[1], outline="#477766")
        d.polygon([p(pt) for pt in [(xx, yy + 12), (xx + 12, yy), (xx + 19, yy + 5), (xx + 7, yy + 17)]], fill=colors[2], outline="#477766")


def fmap(d, x, y, width, height, blue=False):
    edge, pale, bar = ("#6b94bf", "#e6eef8", "#7ea7d3") if blue else (GREEN, "#e2f3e9", "#278664")
    box(d, (x, y, x + width, y + height), pale, edge, 1, 2)
    d.rectangle(p((x + 6, y + 6, x + 12, y + height - 6)), fill=bar)
    if width > 20:
        d.rectangle(p((x + 18, y + 6, x + 24, y + height - 18)), fill="#b1ceec" if blue else "#91d1a9")


def module(d, xy, heading, sub):
    box(d, xy, "#f5fbf8", GREEN, 1, 7)
    txt(d, (xy[0] + 20, xy[1] + 30), heading, 16, GREEN, True)
    txt(d, (xy[0] + 20, xy[1] + 50), sub, 11, MUTED)


def main():
    canvas = Image.new("RGB", (1200 * S, 500 * S), "#ffffff")
    d = ImageDraw.Draw(canvas)
    source = Image.open(SOURCE).convert("RGB")
    txt(d, (24, 31), "v5. Anchor-constrained earring detail recovery", 18, DARK, True)
    txt(d, (1176, 29), "proposed: green   frozen StyleGAN2: blue", 9, "#6b7a82", False, "ra")
    line(d, [(24, 47), (1176, 47)], "#d6e0e3", 1)

    # Two inputs.
    txt(d, (66, 91), "Source I_s", 12, DARK, True, "ma")
    box(d, (22, 101, 110, 189), "#fff", "#33434b")
    photo(canvas, source, (26, 105, 106, 185))
    txt(d, (66, 292), "Blend I_b", 12, DARK, True, "ma")
    box(d, (22, 302, 110, 390), "#fff", "#33434b")
    photo(canvas, source, (26, 306, 106, 386), .6)
    txt(d, (66, 407), "preliminary blend", 9, "#6b7a82", False, "ma")

    # Anchor query.
    module(d, (140, 72, 390, 418), "Anchor-guided localization", "frozen parsing + visible ear")
    arrow(d, [(110, 145), (126, 145), (126, 166), (140, 166)], dash=(5, 4))
    arrow(d, [(110, 346), (126, 346), (126, 190), (140, 190)], dash=(5, 4))
    box(d, (160, 146, 207, 207), "#fff", GREEN, 1, 3)
    txt(d, (183, 166), "parsing", 9, MUTED, False, "ma")
    for x, y, color in [(168, 178, "#e5a27e"), (185, 178, "#76b7c0"), (168, 193, "#9bc98a"), (185, 193, "#ddc86d")]:
        d.rectangle(p((x, y, x + 11, y + 10)), fill=color)
    arrow(d, [(209, 177), (223, 177)])
    box(d, (226, 141, 367, 211), "#fff", GREEN, 1, 3)
    line(d, [(239, 196), (239, 173), (260, 154), (281, 150), (304, 158), (335, 174), (353, 196)], "#8d9ba1", 3)
    circle(d, (246, 193), 6, None, GREEN, 2); circle(d, (346, 193), 6, None, GREEN, 2)
    line(d, [(246, 199), (246, 205)], GREEN, 2); line(d, [(346, 199), (346, 205)], GREEN, 2)
    txt(d, (296, 230), "ear-lobe landmarks", 9, MUTED, False, "ma")
    arrow(d, [(296, 240), (296, 259)])
    fmap(d, 254, 262, 84, 63)
    line(d, [(265, 316), (274, 271), (284, 316), (302, 316), (314, 271), (327, 316)], GREEN, 4)
    txt(d, (296, 348), "ear query Q", 12, GREEN, True, "ma")
    txt(d, (160, 393), "spatially stable ear support", 9, "#6b7a82")

    # HF prior and dynamic mask.
    module(d, (415, 72, 735, 418), "Earring-aware prior", "high-frequency detail + dynamic mask")
    txt(d, (435, 151), "High-frequency extractor", 12, DARK, True)
    box(d, (435, 163, 470, 208), "#263338")
    photo(canvas, source, (439, 167, 466, 204))
    arrow(d, [(472, 185), (482, 185)]); circle(d, (496, 185), 13, "#f0f4de", "#849660"); txt(d, (496, 185), "G", 12, DARK, True, "mm")
    arrow(d, [(510, 185), (520, 185)]); circle(d, (535, 185), 13); txt(d, (535, 185), "-", 13, GREEN, True, "mm")
    arrow(d, [(549, 185), (559, 185)]); plates(d, 561, 168, 2)
    arrow(d, [(607, 185), (617, 185)]); fmap(d, 619, 163, 31, 45)
    txt(d, (635, 225), "P_HF", 12, GREEN, True, "ma")
    txt(d, (545, 241), "I_s - Gsigma(I_s), Conv 3x3", 9, "#6b7a82", False, "ma")
    line(d, [(435, 255), (714, 255)], "#cbded4", 1)
    txt(d, (435, 279), "Recall and fine-mask refinement", 12, DARK, True)
    box(d, (435, 292, 469, 341), "#20272a")
    line(d, [(441, 333), (447, 301), (453, 336), (460, 309), (466, 328)], "#f0a171", 2)
    arrow(d, [(471, 316), (481, 316)]); fmap(d, 484, 292, 32, 49)
    line(d, [(490, 332), (500, 300), (510, 332)], GREEN, 4); txt(d, (500, 358), "M_e", 11, GREEN, True, "ma")
    arrow(d, [(518, 316), (528, 316)]); plates(d, 530, 298, 3)
    arrow(d, [(598, 316), (608, 316)]); box(d, (610, 295, 638, 337), "#fff", GREEN); txt(d, (624, 316), "1x1", 8, GREEN, False, "mm")
    arrow(d, [(640, 316), (650, 316)]); fmap(d, 652, 295, 27, 42)
    line(d, [(658, 330), (665, 301), (673, 330)], GREEN, 3); txt(d, (665, 358), "M_f", 11, GREEN, True, "ma")
    txt(d, (567, 390), "edge / contrast recall; Conv 3x3 + SiLU; sigmoid", 8, "#6b7a82", False, "ma")
    arrow(d, [(338, 293), (400, 293), (400, 185), (415, 185)])
    arrow(d, [(338, 293), (400, 293), (400, 316), (415, 316)])

    # Brightness re-estimation.
    module(d, (755, 72, 930, 418), "Brightness", "re-estimator")
    txt(d, (842, 146), "P_HF, M_f, I_b, Q", 9, MUTED, False, "ma")
    plates(d, 776, 166, 2)
    arrow(d, [(825, 198), (834, 198)]); circle(d, (850, 198), 14); txt(d, (850, 198), "GAP", 8, MUTED, True, "mm")
    arrow(d, [(865, 198), (874, 198)]); box(d, (877, 174, 912, 192), "#fff", GREEN); box(d, (877, 205, 912, 223), "#fff", GREEN)
    txt(d, (894, 187), "FC g", 8, MUTED, False, "mm"); txt(d, (894, 218), "FC b", 8, MUTED, False, "mm")
    txt(d, (842, 270), "P'_HF = g P_HF + b", 12, GREEN, True, "ma")
    fmap(d, 826, 287, 33, 38)
    txt(d, (842, 351), "adapted P'_HF", 12, GREEN, True, "ma")
    arrow(d, [(650, 185), (755, 185)])
    arrow(d, [(679, 316), (720, 316), (720, 225), (755, 225)])

    # Frozen generator and compact shared HFDA.
    box(d, (950, 72, 1176, 418), "#f3f7fb", BLUE, 1, 7)
    txt(d, (970, 102), "Frozen StyleGAN2", 16, "#5c7e9e", True)
    txt(d, (970, 122), "two-scale detail injection", 11, MUTED)
    box(d, (970, 143, 1156, 227), "#fff", GREEN, 1, 4)
    txt(d, (984, 165), "shared HFDA", 12, GREEN, True)
    box(d, (984, 177, 1011, 208), "#fff", GREEN); txt(d, (997, 197), "1x1", 8, GREEN, False, "mm")
    arrow(d, [(1013, 192), (1020, 192)]); circle(d, (1032, 192), 11); txt(d, (1032, 192), "||", 11, DARK, True, "mm")
    arrow(d, [(1044, 192), (1051, 192)]); box(d, (1053, 177, 1084, 208), "#fff", GREEN); txt(d, (1068, 189), "3x3", 8, GREEN, False, "mm"); txt(d, (1068, 202), "1x1", 8, GREEN, False, "mm")
    arrow(d, [(1086, 192), (1093, 192)]); txt(d, (1118, 196), "M_f x dF", 9, GREEN, True, "ma")
    arrow(d, [(930, 306), (950, 306), (950, 192), (970, 192)])
    txt(d, (1062, 248), "shared residuals at 64^2 and 128^2", 8, "#6b7a82", False, "ma")
    box(d, (967, 278, 998, 316), "#fff", BLUE); txt(d, (982, 294), "F64", 8, DARK, False, "ma"); txt(d, (982, 306), "64^2", 8, MUTED, False, "ma")
    arrow(d, [(999, 297), (1006, 297)], "#6c8194"); circle(d, (1020, 297), 12, "#fff", GREEN, 2); txt(d, (1020, 297), "+", 14, GREEN, True, "mm")
    arrow(d, [(1033, 297), (1040, 297)], "#6c8194"); plates(d, 1042, 279, 2, True); txt(d, (1066, 345), "G5", 9, MUTED, False, "ma")
    arrow(d, [(1091, 297), (1098, 297)], "#6c8194"); fmap(d, 1100, 278, 31, 38, True); txt(d, (1115, 332), "F128", 8, MUTED, False, "ma")
    arrow(d, [(1132, 297), (1139, 297)], "#6c8194"); circle(d, (1152, 297), 12, "#fff", GREEN, 2); txt(d, (1152, 297), "+", 14, GREEN, True, "mm")
    txt(d, (1170, 317), "G6-8", 8, MUTED)
    arrow(d, [(1020, 227), (1020, 284)])
    arrow(d, [(1152, 227), (1152, 284)])
    txt(d, (1020, 270), "@64^2", 8, "#6b7a82", False, "ma"); txt(d, (1152, 270), "@128^2", 8, "#6b7a82", False, "ma")
    txt(d, (970, 391), "F' = F + M_f x dF", 9, "#6b7a82")
    txt(d, (1176, 460), "Anchor -> high-frequency prior / fine mask -> brightness adaptation -> shared two-scale HFDA", 9, "#6b7a82", False, "ra")
    canvas.save(OUT, quality=100)
    print(OUT)


if __name__ == "__main__":
    main()
