"""Рендер публикационных кадров AtomicNG → announce/assets/*.png (PIL, без браузера)."""
from __future__ import annotations

import os

from PIL import Image, ImageDraw, ImageFont

FONTS = r"C:\Windows\Fonts"
ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "announce", "assets")
os.makedirs(ASSETS, exist_ok=True)

ACCENT = (74, 108, 247)
ACCENT2 = (0, 200, 160)
BG = (13, 16, 33)
BG2 = (18, 22, 42)
FG = (240, 242, 250)
MUT = (150, 155, 180)
GOOD = (80, 220, 150)


def F(name: str, size: int):
    return ImageFont.truetype(os.path.join(FONTS, name), size)


def seg(size: int, bold=False):
    return F("segoeuib.ttf" if bold else "segoeui.ttf", size)


def mono(size: int, bold=False):
    return F("consolab.ttf" if bold else "consola.ttf", size)


def center_text(draw, cx, y, text, font, fill):
    w = draw.textlength(text, font=font)
    draw.text((cx - w / 2, y), text, font=font, fill=fill)
    return w


def chip(draw, x, y, text, font, fg=FG, bg=ACCENT, pad=14):
    w = draw.textlength(text, font=font)
    h = font.size + pad
    draw.rounded_rectangle([x, y, x + w + 2 * pad, y + h], radius=h // 2, fill=bg)
    draw.text((x + pad, y + pad // 2), text, font=font, fill=fg)
    return w + 2 * pad, h


def watermark(draw_img: Image.Image, text: str, font, xy, color=(74, 108, 247, 26)):
    layer = Image.new("RGBA", draw_img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.text(xy, text, font=font, fill=color)
    draw_img.paste(layer, (0, 0), layer)


# ════════════════════════════════════ 1. HERO ════════════════════════════════════
img = Image.new("RGB", (1200, 675), BG)
d = ImageDraw.Draw(img)
watermark(img, "Δ", F("segoeuib.ttf", 480), (830, 90))
d.text((70, 56), "AtomicNG", font=seg(44, bold=True), fill=FG)
chip(d, 70, 130, "1.0.0 · pip install atomicng", mono(22), fg=(255, 255, 255), bg=ACCENT)

# формула T = T⁰ + Δ (надстрочный 0 вручную)
fx, fy = 70, 250
f_big = mono(96, bold=True)
f_sup = mono(52, bold=True)
d.text((fx, fy), "T = T", font=f_big, fill=FG)
x = fx + d.textlength("T = T", font=f_big)
d.text((x, fy + 8), "0", font=f_sup, fill=ACCENT2)
x += d.textlength("0", font=f_sup)
d.text((x, fy), " + Δ", font=f_big, fill=FG)

d.text((72, 390), "new facts → sparse patch Δ  ·  pretrained table frozen", font=seg(30), fill=MUT)
d.text((72, 436), "clear Δ  →  bit-identical base model", font=seg(30), fill=GOOD)

labels = ["85/85 tests", "bit-exact rollback", "TPU 8.4 ms/iter"]
x = 70
for t in labels:
    w, h = chip(d, x, 520, t, seg(24, bold=True), fg=(255, 255, 255), bg=(30, 38, 70))
    x += w + 24

d.text((70, 610), "github.com/Akseleu-J/atomicng   ·   pypi.org/project/atomicng   ·   Apache-2.0",
       font=mono(20), fill=MUT)
img.save(os.path.join(ASSETS, "hero.png"))

# ════════════════════════════════════ 2. MATRIX ════════════════════════════════════
img = Image.new("RGB", (1200, 780), BG)
d = ImageDraw.Draw(img)
d.text((70, 50), "Live calibration matrix", font=seg(40, bold=True), fill=FG)
d.text((70, 108), "every number — from a real run  ·  CALIBRATION.md", font=seg(22), fill=MUT)

cols = [60, 360, 560, 780]
head_y, row_h = 170, 96
rows = [
    ("Windows CPU", "85/85", "~50 s / group", "baseline"),
    ("Linux CPU (Kaggle)", "72/72", "—", "860 steps → acquisition +0.0422"),
    ("Linux CUDA T4", "72/72", "73 ms / group", "bit-exact Δ rollback"),
    ("Colab TPU-VM CPU", "—", "54 ms / group", "2 training runs, min loss 1.60"),
    ("TPU chip (JAX)", "+", "8.4 ms / iter", "D1 math on chip"),
]
# заголовок
for x, label in zip(cols, ["Environment", "pytest", "BPTT stand", "Training"]):
    d.text((x + 8, head_y), label, font=seg(22, bold=True), fill=ACCENT2)
for i, row in enumerate(rows):
    y = head_y + 46 + i * row_h
    if i % 2 == 0:
        d.rounded_rectangle([60, y - 10, 1140, y + row_h - 18], radius=12, fill=BG2)
    d.text((cols[0] + 8, y), row[0], font=seg(26, bold=True), fill=FG)
    d.text((cols[1] + 8, y), row[1], font=mono(24), fill=GOOD)
    d.text((cols[2] + 8, y), row[2], font=mono(24), fill=FG)
    d.text((cols[3] + 8, y), row[3], font=seg(20), fill=MUT)
d.text((70, head_y + 46 + len(rows) * row_h + 8),
       "writer trained on SmolLM2-135M  ·  corpus: Gutenberg  ·  full data: CALIBRATION.md",
       font=seg(20), fill=MUT)
img.save(os.path.join(ASSETS, "matrix.png"))

# ════════════════════════════════════ 3. DIAGRAM ════════════════════════════════════
img = Image.new("RGB", (1200, 640), BG)
d = ImageDraw.Draw(img)
d.text((70, 50), "How it works", font=seg(40, bold=True), fill=FG)


def box(x, y, w, h, title, sub, border, title_fill=FG):
    d.rounded_rectangle([x, y, x + w, y + h], radius=16, outline=border, width=3, fill=BG2)
    d.text((x + 24, y + 22), title, font=mono(26, bold=True), fill=title_fill)
    d.text((x + 24, y + 62), sub, font=seg(19), fill=MUT)


def arrow(x1, y1, x2, y2, color=ACCENT):
    d.line([x1, y1, x2, y2], fill=color, width=4)
    d.polygon([(x2, y2), (x2 - 16, y2 - 9), (x2 - 16, y2 + 9)], fill=color)


box(70, 170, 300, 110, "T0  frozen", "pretrained n-gram table", (150, 155, 180))
d.text((390, 205), "+", font=mono(48, bold=True), fill=FG)
box(450, 170, 300, 110, "Δ  sparse patch", "new facts (data-only)", ACCENT, title_fill=ACCENT2)
arrow(770, 225, 850, 225)
box(855, 170, 280, 110, "D1  write", "convex · bounded", ACCENT)

arrow(430, 330, 430, 400, ACCENT2)
d.text((450, 345), "writer: targets + gates  (forward-only)", font=seg(22), fill=FG)

box(70, 405, 480, 110, "Model reads  T = T0 + Δ", "before layer-2 attention", ACCENT2)
d.text((600, 528), "clear Δ  →  byte-identical base model", font=seg(24, bold=True), fill=GOOD)

d.text((70, 555), "D1:  A_j = mean(β·v)   ·   Δ_j⁺ = (1−w)·Δ_j + A_j   ·   convex → bounded",
       font=mono(24), fill=MUT)
img.save(os.path.join(ASSETS, "diagram.png"))

# ════════════════════════════════════ 4. QUICKSTART ════════════════════════════════════
img = Image.new("RGB", (1200, 675), BG)
d = ImageDraw.Draw(img)
card_x, card_y, card_w, card_h = 60, 60, 1080, 555
d.rounded_rectangle([card_x, card_y, card_x + card_w, card_y + card_h], radius=18, fill=(22, 26, 48))
for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
    d.ellipse([card_x + 28 + i * 34, card_y + 24, card_x + 44 + i * 34, card_y + 40], fill=c)
d.text((card_x + card_w - 340, card_y + 22), "quickstart · CPU only", font=mono(20), fill=MUT)

code = [
    ("$ pip install atomicng", GOOD),
    ("", FG),
    ("import atomicng", FG),
    ("", FG),
    ("model = atomicng.TinyModel()", FG),
    ("atomicng.write(model, \"New fact about kangaroos.\")", FG),
    ("atomicng.verify_deletion(model)", FG),
    ("# clear Δ → byte-identical to base — guaranteed", MUT),
]
y = card_y + 70
for line, color in code:
    d.text((card_x + 40, y), line, font=mono(27), fill=color)
    y += 46
d.text((70, 628), "no GPU needed  ·  core is pure NumPy  ·  5 seconds to first write",
       font=seg(22), fill=MUT)
img.save(os.path.join(ASSETS, "quickstart.png"))

print("rendered:", sorted(os.listdir(ASSETS)))
