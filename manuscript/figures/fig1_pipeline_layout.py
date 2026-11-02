"""Rough Figure 1 layout mockup using the supplied pipeline artwork."""
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

SRC = "/mnt/c/Users/letgao/.dsh/attachments/v1/objects/06/06ae3674912fe07741b6fc67ceca4276dc2b658bbf625eaa82f9e8d2a1d3b01d"
OUT = Path("/home/glt/rnaxgb/overleaf/figures/fig1_pipeline_layout.png")
W, H = 1830, 1750
img = Image.new("RGB", (W, H), "#FAFAF7")
d = ImageDraw.Draw(img)
INK = "#14213D"; TEAL = "#2A9D8F"; AMBER = "#E9C46A"; CORAL = "#E76F51"; GREY = "#B8B8B8"

def font(size, bold=False):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
                                  else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()

# top pipeline
src = Image.open(SRC).convert("RGBA")
bg = Image.new("RGBA", src.size, "white")
bg.alpha_composite(src)
src = bg.convert("RGB")
top_h = int(src.height * W / src.width)
src = src.resize((W, top_h), Image.LANCZOS)
img.paste(src, (0, 0))
d.text((18, 10), "a", fill=INK, font=font(44, True))

# bottom cards
def card(x, y, w, h, title, lines, accent=TEAL):
    d.rounded_rectangle((x, y, x+w, y+h), radius=10, fill="white", outline=GREY, width=2)
    d.text((x+22, y+14), title, fill=INK, font=font(30, True))
    yy = y + 70
    for line in lines:
        d.text((x+22, yy), line, fill="#444444", font=font(24))
        yy += 36
    d.line((x+20, y+h-8, x+w-20, y+h-8), fill=accent, width=4)

card(40, 1010, 850, 430, "b  Recovery by sequence population",
     ["All 64.6%", "Exact repeats 87.1%", "Sequence-novel 55.1%", "<80% identity 48.9%",
      "Separate training evaluation: 70.0 / 50.4"], TEAL)
card(940, 1010, 850, 430, "c  Confidence-threshold transport",
     ["Validation 51.9 / 90.1", "Repeats 79.9 / 95.1", "Non-repeats 28.0 / 80.8",
      "same threshold, different population"], AMBER)

# d evidence gates
d.rounded_rectangle((40, 1480, 1790, 1715), radius=10, fill="white", outline=GREY, width=2)
d.text((60, 1495), "d  Evidence types for a design claim", fill=INK, font=font(30, True))
gates = [("compatibility", "native vs decoy"), ("forward model", "AF3 consistency"),
         ("experiment", "structure / function")]
for i, (a, b) in enumerate(gates):
    x = 95 + i * 560
    d.rounded_rectangle((x, 1550, x+420, 1665), radius=8, fill="#FFFFFF",
                        outline=[TEAL, AMBER, CORAL][i], width=3)
    d.text((x+18, 1570), a, fill=INK, font=font(26, True))
    d.text((x+18, 1612), b, fill="#555555", font=font(22))
    if i < 2:
        d.line((x+420, 1607, x+470, 1607), fill=INK, width=4)
        d.polygon([(x+470, 1607), (x+455, 1598), (x+455, 1616)], fill=INK)
d.text((640, 1680), "not interchangeable", fill=CORAL, font=font(24, True))

img.save(OUT, quality=95)
print("wrote", OUT, img.size)
