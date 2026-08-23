from PIL import Image, ImageDraw
import io
from utils.ela_processor import calculate_ela, get_ela_stats, generate_forgery_score

# 1. Clean Authentic JPEG (single compression cycle)
base = Image.new("RGB", (400, 300), color=(140, 180, 220))
draw = ImageDraw.Draw(base)
# Draw smooth mountains and sun
draw.ellipse([50, 30, 110, 90], fill=(255, 230, 100))
draw.polygon([(0, 300), (150, 120), (300, 300)], fill=(70, 120, 70))
draw.polygon([(180, 300), (320, 160), (400, 300)], fill=(50, 90, 50))

buf = io.BytesIO()
base.save(buf, format="JPEG", quality=90)
buf.seek(0)
authentic_jpg = Image.open(buf).convert("RGB")
authentic_jpg.save("samples/eval_results/clean_authentic.jpg", format="JPEG", quality=90)

# 2. Spliced Altered Image: Insert a textured patch with different compression history
donor = Image.new("RGB", (120, 100), color=(200, 50, 50))
d_draw = ImageDraw.Draw(donor)
for i in range(0, 120, 10):
    d_draw.line([(i, 0), (i, 100)], fill=(255, 255, 255), width=3)

buf_d = io.BytesIO()
donor.save(buf_d, format="JPEG", quality=50) # Heavy compression
buf_d.seek(0)
donor_jpg = Image.open(buf_d).convert("RGB")

spliced = authentic_jpg.copy()
spliced.paste(donor_jpg, (200, 80)) # Spliced in the sky/mountain area
spliced.save("samples/eval_results/clean_spliced.jpg", format="JPEG", quality=90)

# Evaluate both
ela_auth = calculate_ela(authentic_jpg, quality=90, amplify=20.0)
s_auth, l_auth = generate_forgery_score(ela_auth)
stats_auth = get_ela_stats(ela_auth)

ela_splice = calculate_ela(spliced, quality=90, amplify=20.0)
s_splice, l_splice = generate_forgery_score(ela_splice)
stats_splice = get_ela_stats(ela_splice)

print(f"Clean Authentic Image  -> Score: {s_auth:3d}% | Label: {l_auth:10s} | Variance: {stats_auth['variance']:6.2f} | HighPct: {stats_auth['high_pct']:.1f}%")
print(f"Altered Spliced Image  -> Score: {s_splice:3d}% | Label: {l_splice:10s} | Variance: {stats_splice['variance']:6.2f} | HighPct: {stats_splice['high_pct']:.1f}%")
