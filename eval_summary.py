from PIL import Image
from utils.ela_processor import calculate_ela, get_ela_stats, generate_forgery_score

tests = [
    ('test img 1.webp', 'User Test 1 (Zebra donor)'),
    ('test img2.webp', 'User Test 2 (3-panel splicing figure)'),
    ('samples/eval_results/panel_a_authentic.png', 'Panel A (Authentic landscape)'),
    ('samples/eval_results/panel_c_spliced.png', 'Panel C (Spliced composite)'),
    ('samples/eval_results/synthetic_spliced.jpg', 'Altered Image (Zebra spliced onto landscape)'),
]

for path, desc in tests:
    img = Image.open(path).convert('RGB')
    ela = calculate_ela(img, quality=90, amplify=20.0)
    score, label = generate_forgery_score(ela)
    stats = get_ela_stats(ela)
    print(f"[{label.upper():10s}] {score:3d}%  |  Mean: {stats['mean']:5.2f}  |  Var: {stats['variance']:7.1f}  |  {desc}")
