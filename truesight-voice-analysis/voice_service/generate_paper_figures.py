"""
generate_paper_figures.py

Generates publication-quality, crisp, light-background IEEE conference figures:
1. fig1_architecture.png: End-to-end TRUESIGHT system architecture diagram
2. fig2_feature_pipeline.png: 42-feature acoustic extraction & VAD gating pipeline
3. fig3_confusion_baseline.png: Baseline Linear Model Confusion Matrix (CBU5201 N=100)
4. fig4_confusion_production.png: Production Random Forest Confusion Matrix (CBU5201 N=100)
5. fig5_model_comparison.png: Cross-model performance comparison chart
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FIG_DIR = os.path.join(os.path.dirname(__file__), "research_paper", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def create_architecture_diagram():
    w, h = 1200, 520
    img = Image.new("RGB", (w, h), color="#ffffff")
    draw = ImageDraw.Draw(img)

    # Outer border
    draw.rectangle([10, 10, w - 10, h - 10], outline="#cccccc", width=1)

    # Title Banner
    draw.rectangle([20, 20, w - 20, 60], fill="#f5f7fa", outline="#d1d5db", width=1)
    draw.text((40, 32), "Fig. 1. Architectural schematic of the TRUESIGHT Voice Analysis Pipeline", fill="#1f2937")

    # Pipeline Boxes
    boxes = [
        ("1. Audio Ingestion", ["* File Upload (.wav, .mp3, .m4a)", "* Browser WebRTC Microphone", "* Streaming MediaRecorder chunks"], 40, 100, 200, 380, "#eef2ff", "#4f46e5"),
        ("2. Signal Preprocessing", ["* Mono channel folding", "* 16 kHz sinc resampling", "* RMS peak normalization", "* VAD voice-presence gating", "* SNR & clipping audits"], 260, 100, 200, 380, "#f0fdf4", "#16a34a"),
        ("3. 42-D Feature Extraction", ["* Praat F0 pitch (mean, std)", "* Jitter & Shimmer ratios", "* Silence TextGrid dynamics", "* 13 MFCC means & stds", "* Spectral centroid & rolloff"], 480, 100, 220, 380, "#fefce8", "#ca8a04"),
        ("4. ML Inference Engine", ["* Serialized Random Forest", "* 200 ensemble decision trees", "* Output: P(Deceptive) in [0, 1]", "* Fixed 0.50 threshold", "* Nominal verdict generation"], 720, 100, 210, 380, "#eff6ff", "#2563eb"),
        ("5. Explanation & API", ["* Rule-based observation heuristics", "* Signal quality warnings", "* Scientific ethical caveats", "* FastAPI REST endpoints", "* Real-time WebSocket feed"], 950, 100, 210, 380, "#faf5ff", "#9333ea"),
    ]

    for title, items, x, y, bw, bh, bg, border in boxes:
        draw.rectangle([x, y, x + bw, y + bh], fill=bg, outline=border, width=2)
        draw.rectangle([x, y, x + bw, y + 40], fill=border)
        draw.text((x + 12, y + 12), title, fill="#ffffff")

        cur_y = y + 55
        for item in items:
            draw.text((x + 12, cur_y), item, fill="#1f2937")
            cur_y += 32

    # Draw Connecting Arrows
    arrow_y = 250
    for i in range(len(boxes) - 1):
        x1 = boxes[i][2] + boxes[i][4]
        x2 = boxes[i + 1][2]
        draw.line([x1 + 4, arrow_y, x2 - 4, arrow_y], fill="#4b5563", width=3)
        # arrowhead
        draw.polygon([(x2 - 4, arrow_y), (x2 - 12, arrow_y - 6), (x2 - 12, arrow_y + 6)], fill="#4b5563")

    img.save(os.path.join(FIG_DIR, "fig1_architecture.png"))
    print("Created fig1_architecture.png")


def create_feature_pipeline_diagram():
    w, h = 1100, 480
    img = Image.new("RGB", (w, h), color="#ffffff")
    draw = ImageDraw.Draw(img)

    draw.rectangle([10, 10, w - 10, h - 10], outline="#cccccc", width=1)
    draw.rectangle([20, 20, w - 20, 60], fill="#f5f7fa", outline="#d1d5db", width=1)
    draw.text((40, 32), "Fig. 2. Dual-Engine 42-Dimensional Acoustic Feature Extraction and Gating Workflow", fill="#1f2937")

    # Waveform Input
    draw.rectangle([40, 100, 220, 200], fill="#e0f2fe", outline="#0284c7", width=2)
    draw.text((55, 115), "Raw Audio Waveform", fill="#0369a1")
    draw.text((55, 145), "16 kHz Mono Signal\nDuration >= 0.50 s", fill="#334155")

    # VAD Gate
    draw.polygon([(130, 250), (220, 310), (130, 370), (40, 310)], fill="#fee2e2", outline="#dc2626")
    draw.text((75, 300), "VAD Voice Gate\nEnergy > -45 dB", fill="#991b1b")

    draw.line([130, 200, 130, 250], fill="#0284c7", width=3)

    # Sub-engines
    draw.rectangle([340, 100, 660, 260], fill="#fef3c7", outline="#d97706", width=2)
    draw.text((360, 115), "Engine A: Praat-Parselmouth (10 Prosodic Features)", fill="#b45309")
    draw.text((360, 145), "* F0 Pitch: Mean, Std, Min, Max (75-500 Hz)\n* Jitter Local (% glottal period deviation, >=25 cycles)\n* Shimmer Local (% amplitude perturbation, >=25 cycles)\n* Temporal Silences: Pause Count, Pause Ratio, Silence Duration", fill="#451a03")

    draw.rectangle([340, 290, 660, 440], fill="#e0e7ff", outline="#4338ca", width=2)
    draw.text((360, 305), "Engine B: Librosa (32 Spectral & Cepstral Features)", fill="#3730a3")
    draw.text((360, 335), "* Short-Time RMS Energy: Mean & Std\n* Zero-Crossing Rate (ZCR)\n* Spectral Centroid, Bandwidth, and Rolloff (85%)\n* MFCCs 1-13 Means (13 dims) & MFCCs 1-13 Stds (13 dims)", fill="#1e1b4b")

    # Connect VAD to Engines
    draw.line([220, 310, 300, 310], fill="#4b5563", width=2)
    draw.line([300, 310, 300, 180], fill="#4b5563", width=2)
    draw.line([300, 180, 340, 180], fill="#4b5563", width=2)
    draw.line([300, 310, 300, 365], fill="#4b5563", width=2)
    draw.line([300, 365, 340, 365], fill="#4b5563", width=2)

    # Concatenation Block
    draw.rectangle([760, 180, 1040, 360], fill="#ecfdf5", outline="#059669", width=2)
    draw.text((780, 205), "Vector Concatenation (to_vector)", fill="#065f46")
    draw.text((780, 240), "[10 Praat + 6 Spectral + 26 MFCC]\n= 42-Dimensional Feature Vector\n\nCanonical ordering enforced across\ntraining and runtime inference.", fill="#064e3b")

    draw.line([660, 180, 710, 180], fill="#4b5563", width=2)
    draw.line([710, 180, 710, 270], fill="#4b5563", width=2)
    draw.line([710, 270, 760, 270], fill="#4b5563", width=2)

    draw.line([660, 365, 710, 365], fill="#4b5563", width=2)
    draw.line([710, 365, 710, 270], fill="#4b5563", width=2)

    img.save(os.path.join(FIG_DIR, "fig2_feature_pipeline.png"))
    print("Created fig2_feature_pipeline.png")


def create_confusion_matrix_figure(tn, fp, fn, tp, title, subtitle, filename):
    w, h = 600, 520
    img = Image.new("RGB", (w, h), color="#ffffff")
    draw = ImageDraw.Draw(img)

    draw.rectangle([10, 10, w - 10, h - 10], outline="#cccccc", width=1)
    draw.text((30, 25), title, fill="#111827")
    draw.text((30, 50), subtitle, fill="#4b5563")

    top_y = 110
    left_x = 160
    cell_size = 170

    # Headers
    draw.text((left_x + 35, top_y - 25), "Predicted: Truth", fill="#374151")
    draw.text((left_x + cell_size + 25, top_y - 25), "Predicted: Deceptive", fill="#374151")

    draw.text((25, top_y + 70), "Actual: Truth", fill="#374151")
    draw.text((25, top_y + cell_size + 70), "Actual: Deceptive", fill="#374151")

    cells = [
        (0, 0, tn, "TN", "#dcfce7", "#16a34a"),
        (0, 1, fp, "FP", "#fee2e2", "#dc2626"),
        (1, 0, fn, "FN", "#fee2e2", "#dc2626"),
        (1, 1, tp, "TP", "#dcfce7", "#16a34a"),
    ]

    total = tn + fp + fn + tp
    for r, c, val, tag, bg, border in cells:
        x0 = left_x + c * cell_size
        y0 = top_y + r * cell_size
        x1 = x0 + cell_size
        y1 = y0 + cell_size

        draw.rectangle([x0, y0, x1, y1], fill=bg, outline=border, width=2)
        pct = (val / total) * 100
        draw.text((x0 + 55, y0 + 60), f"{tag}: {val}", fill="#111827")
        draw.text((x0 + 65, y0 + 90), f"({pct:.1f}%)", fill="#4b5563")

    # Metrics summary at bottom
    acc = (tp + tn) / total
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    draw.rectangle([30, 465, w - 30, 500], fill="#f9fafb", outline="#e5e7eb")
    draw.text((45, 475), f"Accuracy: {acc:.1%}  |  Sensitivity (Deceptive): {sens:.1%}  |  Specificity: {spec:.1%}", fill="#1f2937")

    img.save(os.path.join(FIG_DIR, filename))
    print(f"Created {filename}")


def create_model_comparison_chart():
    w, h = 900, 500
    img = Image.new("RGB", (w, h), color="#ffffff")
    draw = ImageDraw.Draw(img)

    draw.rectangle([10, 10, w - 10, h - 10], outline="#cccccc", width=1)
    draw.text((30, 25), "Fig. 5. Empirical Performance Comparison on Unseen External CBU5201 Benchmark (N=100)", fill="#111827")
    draw.text((30, 50), "Comparison of Standardized Linear Baseline vs. 200-Tree Production Random Forest", fill="#4b5563")

    # Chart Area
    metrics = [
        ("Accuracy", 0.48, 0.56, "+8.0%"),
        ("Balanced Acc", 0.48, 0.56, "+8.0%"),
        ("Macro F1", 0.403, 0.560, "+0.157"),
        ("ROC-AUC", 0.492, 0.613, "+0.121"),
        ("English Acc (N=50)", 0.44, 0.62, "+18.0%"),
        ("Chinese Acc (N=50)", 0.52, 0.50, "-2.0%"),
    ]

    base_y = 100
    row_h = 55
    bar_start_x = 220
    max_w = 400

    for i, (name, b_val, p_val, delta) in enumerate(metrics):
        y = base_y + i * row_h
        draw.text((30, y + 10), name, fill="#1f2937")

        # Baseline bar
        bw = int(b_val * max_w)
        draw.rectangle([bar_start_x, y, bar_start_x + bw, y + 16], fill="#9ca3af", outline="#6b7280")
        draw.text((bar_start_x + bw + 8, y), f"{b_val:.1%}" if "F1" not in name and "AUC" not in name else f"{b_val:.3f}", fill="#4b5563")

        # Production bar
        pw = int(p_val * max_w)
        bar_col = "#2563eb"
        draw.rectangle([bar_start_x, y + 20, bar_start_x + pw, y + 36], fill=bar_col, outline="#1d4ed8")
        draw.text((bar_start_x + pw + 8, y + 20), f"{p_val:.1%}" if "F1" not in name and "AUC" not in name else f"{p_val:.3f}", fill="#1e40af")

        # Delta badge
        delta_col = "#15803d" if delta.startswith("+") else "#b91c1c"
        draw.rectangle([760, y + 8, 860, y + 32], fill="#f3f4f6", outline=delta_col)
        draw.text((775, y + 12), f"Delta: {delta}", fill=delta_col)

    # Legend
    draw.rectangle([bar_start_x, 440, bar_start_x + 18, 456], fill="#9ca3af", outline="#6b7280")
    draw.text((bar_start_x + 26, 440), "Previous Baseline (Linear Logistic Regression)", fill="#374151")

    draw.rectangle([bar_start_x + 320, 440, bar_start_x + 338, 456], fill="#2563eb", outline="#1d4ed8")
    draw.text((bar_start_x + 346, 440), "Current Model (Production Random Forest)", fill="#374151")

    img.save(os.path.join(FIG_DIR, "fig5_model_comparison.png"))
    print("Created fig5_model_comparison.png")


if __name__ == "__main__":
    create_architecture_diagram()
    create_feature_pipeline_diagram()
    create_confusion_matrix_figure(42, 8, 44, 6, "Fig. 3. Baseline Linear Model Confusion Matrix", "Evaluated on CBU5201 External Benchmark (N=100)", "fig3_confusion_baseline.png")
    create_confusion_matrix_figure(29, 21, 23, 27, "Fig. 4. Production Random Forest Confusion Matrix", "Evaluated on CBU5201 External Benchmark (N=100)", "fig4_confusion_production.png")
    create_model_comparison_chart()
