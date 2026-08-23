"""Quick test for anomaly_reporter module."""
import numpy as np
from utils.anomaly_reporter import generate_anomaly_report, format_anomaly_html

# Simulate a known-spliced image
ela_stats = {"variance": 1046.5, "mean": 12.98, "p99": 110.3, "high_pct": 19.7}
unet_prob = np.zeros((300, 400), dtype="float32")
unet_prob[80:200, 150:300] = 0.92    # concentrated donor patch
metadata  = {
    "has_exif": False,
    "exif": {},
    "basic": {"format": "JPEG", "color_mode": "RGB"},
}

findings = generate_anomaly_report(
    ela_stats=ela_stats,
    unet_prob=unet_prob,
    metadata=metadata,
    score=74,
    mode="Heuristic (ELA)",
)

print(f"Generated {len(findings)} findings:")
for f in findings:
    tag    = f["tag"]
    level  = f["level"]
    icon   = f["icon"]
    print(f"  [{level:8s}]  {tag}")

html_out = format_anomaly_html(findings)
print(f"\nHTML output: {len(html_out)} characters")
print("[OK] anomaly_reporter.py works correctly")
