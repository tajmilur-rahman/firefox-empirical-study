#!/usr/bin/env python3
"""
Priority Cluster Analysis — Bug Level with Outlier Detection
- 125 points, one per bug
- Each bug position = mean embedding of all its labels
- Outliers detected per priority using IQR method
- Core bugs = filled circles, outliers = X markers with bug ID
- Ellipses drawn on core points only

Input:
    - priority_bugs_labeled_claude.csv
    - priority_gemini_canonical_with_weight.csv

Output:
    - priority_bug_cluster_interactive.html
    - priority_bug_cluster_static.png

Usage:
    pip install umap-learn matplotlib sentence-transformers plotly pandas openpyxl scikit-learn
    python priority_bug_cluster_analysis.py
"""

import sys
import csv
import warnings

warnings.filterwarnings("ignore")

csv.field_size_limit(10 * 1024 * 1024)


def install(pkg):
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])


try:
    import numpy as np
except ImportError:
    install("numpy");
    import numpy as np

try:
    import pandas as pd
except ImportError:
    install("pandas openpyxl");
    import pandas as pd

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
except ImportError:
    install("matplotlib")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

try:
    import plotly.graph_objects as go
except ImportError:
    install("plotly");
    import plotly.graph_objects as go

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    install("sentence-transformers")
    from sentence_transformers import SentenceTransformer

try:
    import umap
except ImportError:
    install("umap-learn");
    import umap

try:
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import silhouette_score, davies_bouldin_score
except ImportError:
    install("scikit-learn")
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import silhouette_score, davies_bouldin_score

# ── Config ────────────────────────────────────────────────────────────────────
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LABELED_CSV = os.path.join(BASE_DIR, "priority_bugs_labeled_claude.csv")
CLUSTERS_FILE = os.path.join(BASE_DIR, "priority_gemini_canonical_with_weight.csv")
OUTPUT_HTML = os.path.join(BASE_DIR, "mac_priority_bug_cluster_interactive.html")
OUTPUT_PNG = os.path.join(BASE_DIR, "mac_priority_bug_cluster_static.png")
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 256
NOISE_LABELS = {"Unlabeled"}
TARGET_WEIGHT = 0.45
ELLIPSE_STD = 1.5
IQR_FACTOR = 2.0

COLORS = {
    "P1": "#E74C3C",
    "P2": "#E67E22",
    "P3": "#3498DB",
    "P4": "#2ECC71",
    "P5": "#9B59B6",
}


# ─────────────────────────────────────────────────────────────────────────────

def load_bugs(labeled_csv):
    bugs = []
    with open(labeled_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels = [
                l.strip() for l in row['matched_labels'].split(';')
                if l.strip() and l.strip() not in NOISE_LABELS
            ]
            if labels:
                bugs.append({
                    'bug_id': row['bug_id'],
                    'priority': row['priority'],
                    'labels': labels,
                    'label_count': len(labels)
                })
    print(f"  Loaded {len(bugs)} bugs")
    for pri in ['P1', 'P2', 'P3', 'P4', 'P5']:
        count = sum(1 for b in bugs if b['priority'] == pri)
        print(f"    {pri}: {count} bugs")
    return bugs


def load_weights(clusters_file):
    weights = {}
    with open(clusters_file, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            weights[row["canonical_phrase"].strip()] = float(row["weight"])
    return weights


def detect_outliers_iqr(x, y, factor=1.5):
    x, y = np.array(x), np.array(y)
    cx, cy = np.mean(x), np.mean(y)
    distances = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    q1 = np.percentile(distances, 25)
    q3 = np.percentile(distances, 75)
    iqr = q3 - q1
    threshold = q3 + factor * iqr
    return distances <= threshold


def draw_confidence_ellipse(ax, x, y, color, n_std=1.5, alpha=0.15):
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    width = 2 * n_std * np.sqrt(abs(eigenvalues[0]))
    height = 2 * n_std * np.sqrt(abs(eigenvalues[1]))
    ellipse = Ellipse(
        xy=(np.mean(x), np.mean(y)),
        width=width, height=height, angle=angle,
        facecolor=color, alpha=alpha,
        edgecolor=color, linewidth=2, linestyle="--"
    )
    ax.add_patch(ellipse)


def run():
    # Fix random seeds for cross-platform reproducibility
    import random, os
    random.seed(42)
    np.random.seed(42)
    os.environ['PYTHONHASHSEED'] = '42'

    # Step 1
    print("Step 1: Loading bug data...")
    bugs = load_bugs(LABELED_CSV)
    weights = load_weights(CLUSTERS_FILE)

    # Step 2
    print(f"\nStep 2: Collecting unique labels...")
    all_unique_labels = sorted(list(set(label for bug in bugs for label in bug['labels'])))
    print(f"  Total unique labels: {len(all_unique_labels)}")

    # Step 3
    print(f"\nStep 3: Embedding labels with {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    label_embeddings = model.encode(
        all_unique_labels,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        show_progress_bar=True
    )
    label_to_emb = {label: label_embeddings[i] for i, label in enumerate(all_unique_labels)}
    print(f"  Embedding shape: {label_embeddings.shape}")

    # Step 4
    print(f"\nStep 4: Computing mean embedding per bug...")
    bug_embeddings = []
    valid_bugs = []
    for bug in bugs:
        embs = [label_to_emb[l] for l in bug['labels'] if l in label_to_emb]
        if embs:
            bug_embeddings.append(np.mean(embs, axis=0))
            valid_bugs.append(bug)

    bug_embeddings = np.array(bug_embeddings)
    priorities = [b['priority'] for b in valid_bugs]
    print(f"  Bug embedding matrix: {bug_embeddings.shape}")

    # Step 5
    print(f"\nStep 5: Reducing to 2D with Supervised UMAP (target_weight={TARGET_WEIGHT})...")
    le = LabelEncoder()
    priority_encoded = le.fit_transform(priorities)
    reducer = umap.UMAP(
        n_components=2, n_neighbors=10,
        min_dist=0.1, metric="cosine",
        target_metric="categorical",
        target_weight=TARGET_WEIGHT,
        random_state=42,
        n_jobs=1
    )
    emb_2d = reducer.fit_transform(bug_embeddings, y=priority_encoded)
    print(f"  2D shape: {emb_2d.shape}")

    # Step 6 — Metrics
    print(f"\nStep 6: Computing cluster quality metrics...")
    from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                                 v_measure_score, homogeneity_score, completeness_score)
    from sklearn.cluster import KMeans

    sil_score = silhouette_score(emb_2d, priorities)
    db_score = davies_bouldin_score(emb_2d, priorities)

    # Use KMeans to get predicted cluster labels from 2D positions
    # then compare with true priority labels
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    pred_labels = kmeans.fit_predict(emb_2d)

    ari_score = adjusted_rand_score(priorities, pred_labels)
    nmi_score = normalized_mutual_info_score(priorities, pred_labels)
    h_score = homogeneity_score(priorities, pred_labels)
    c_score = completeness_score(priorities, pred_labels)
    v_score = v_measure_score(priorities, pred_labels)

    print(f"  Silhouette Score:          {sil_score:.4f} (higher=better, max=1.0)")
    print(f"  Davies-Bouldin Index:      {db_score:.4f}  (lower=better, min=0.0)")
    print(f"  Adjusted Rand Index:       {ari_score:.4f} (higher=better, max=1.0)")
    print(f"  Normalized Mutual Info:    {nmi_score:.4f} (higher=better, max=1.0)")
    print(f"  Homogeneity:               {h_score:.4f}  (higher=better, max=1.0)")
    print(f"  Completeness:              {c_score:.4f}  (higher=better, max=1.0)")
    print(f"  V-Measure:                 {v_score:.4f}  (higher=better, max=1.0)")

    # Group by priority
    pri_data = {pri: {"x": [], "y": [], "bug_ids": [], "labels": [], "label_counts": []} for pri in COLORS}
    for i, bug in enumerate(valid_bugs):
        pri = bug['priority']
        pri_data[pri]["x"].append(emb_2d[i][0])
        pri_data[pri]["y"].append(emb_2d[i][1])
        pri_data[pri]["bug_ids"].append(bug['bug_id'])
        pri_data[pri]["labels"].append(bug['labels'])
        pri_data[pri]["label_counts"].append(bug['label_count'])

    # Step 6b — Outlier detection
    print(f"\nStep 6b: Detecting outliers per priority (IQR factor={IQR_FACTOR})...")
    pri_outlier_mask = {}
    for pri, data in pri_data.items():
        if len(data['x']) >= 4:
            mask = detect_outliers_iqr(data['x'], data['y'], IQR_FACTOR)
            pri_outlier_mask[pri] = mask
            n_out = sum(~mask)
            print(f"  {pri}: {sum(mask)} core bugs, {n_out} outliers")
            if n_out > 0:
                for i, is_core in enumerate(mask):
                    if not is_core:
                        print(f"    Outlier: bug {data['bug_ids'][i]} at ({data['x'][i]:.2f},{data['y'][i]:.2f})")
        else:
            pri_outlier_mask[pri] = np.ones(len(data['x']), dtype=bool)

    # Summary stats
    print(f"\n  Per-Priority Stats:")
    print(
        f"  {'Priority':<10} {'N':>4} {'Core':>6} {'Outliers':>9} {'Centroid X':>12} {'Centroid Y':>12} {'Spread':>8}")
    print(f"  {'-' * 65}")
    centroids = {}
    for pri in ['P1', 'P2', 'P3', 'P4', 'P5']:
        data = pri_data[pri]
        if data['x']:
            mask = pri_outlier_mask[pri]
            cx = np.mean(data['x'])
            cy = np.mean(data['y'])
            spread = np.mean([np.sqrt((x - cx) ** 2 + (y - cy) ** 2) for x, y in zip(data['x'], data['y'])])
            centroids[pri] = (cx, cy)
            print(
                f"  {pri:<10} {len(data['x']):>4} {sum(mask):>6} {sum(~mask):>9} {cx:>12.2f} {cy:>12.2f} {spread:>8.2f}")

    print(f"\n  Pairwise centroid distances:")
    pris = ['P1', 'P2', 'P3', 'P4', 'P5']
    print(f"  {'':12}", end="")
    for p in pris: print(f"{p:>10}", end="")
    print()
    for p1 in pris:
        print(f"  {p1:<12}", end="")
        for p2 in pris:
            d = np.sqrt((centroids[p1][0] - centroids[p2][0]) ** 2 + (centroids[p1][1] - centroids[p2][1]) ** 2)
            print(f"{d:>10.2f}", end="")
        print()

    # ── OPTION 1: Interactive Plotly ──────────────────────────────────────────
    print(f"\nStep 7a: Creating interactive Plotly plot...")
    fig = go.Figure()

    for pri, data in pri_data.items():
        if not data["x"]:
            continue
        mask = pri_outlier_mask[pri]

        core_x = [data['x'][i] for i in range(len(data['x'])) if mask[i]]
        core_y = [data['y'][i] for i in range(len(data['y'])) if mask[i]]
        core_ids = [data['bug_ids'][i] for i in range(len(data['bug_ids'])) if mask[i]]
        core_labels = [data['labels'][i] for i in range(len(data['labels'])) if mask[i]]
        core_counts = [data['label_counts'][i] for i in range(len(data['label_counts'])) if mask[i]]

        out_x = [data['x'][i] for i in range(len(data['x'])) if not mask[i]]
        out_y = [data['y'][i] for i in range(len(data['y'])) if not mask[i]]
        out_ids = [data['bug_ids'][i] for i in range(len(data['bug_ids'])) if not mask[i]]
        out_labels = [data['labels'][i] for i in range(len(data['labels'])) if not mask[i]]
        out_counts = [data['label_counts'][i] for i in range(len(data['label_counts'])) if not mask[i]]

        if core_x:
            sizes = [max(8, min(25, c * 1.2)) for c in core_counts]
            hover = [
                f"<b>Bug {bid}</b> [CORE]<br>Priority: {pri}<br>Labels ({cnt}):<br>" +
                "<br>".join([f"  • {l}" for l in lbls[:5]])
                for bid, lbls, cnt in zip(core_ids, core_labels, core_counts)
            ]
            fig.add_trace(go.Scatter(
                x=core_x, y=core_y, mode="markers",
                name=f"{pri} core (n={len(core_x)})",
                marker=dict(color=COLORS[pri], size=sizes, opacity=0.8,
                            symbol="circle", line=dict(color="white", width=1)),
                text=hover, hoverinfo="text",
                hovertemplate="%{text}<extra></extra>"
            ))

        if out_x:
            hover_out = [
                f"<b>Bug {bid}</b> [OUTLIER]<br>Priority: {pri}<br>Labels ({cnt}):<br>" +
                "<br>".join([f"  • {l}" for l in lbls])  # ALL labels shown
                for bid, lbls, cnt in zip(out_ids, out_labels, out_counts)
            ]
            fig.add_trace(go.Scatter(
                x=out_x, y=out_y, mode="markers+text",
                name=f"{pri} outliers (n={len(out_x)})",
                marker=dict(color=COLORS[pri], size=12, opacity=0.9,
                            symbol="x", line=dict(color=COLORS[pri], width=2)),
                text=[f"  {bid}" for bid in out_ids],
                textposition="middle right",
                textfont=dict(size=9, color=COLORS[pri]),
                hovertext=hover_out, hoverinfo="text",
                hovertemplate="%{hovertext}<extra></extra>"
            ))

        cx, cy = np.mean(data["x"]), np.mean(data["y"])
        fig.add_trace(go.Scatter(
            x=[cx], y=[cy], mode="text",
            text=[f"<b>{pri}</b>"],
            textfont=dict(size=16, color=COLORS[pri]),
            showlegend=False, hoverinfo="skip"
        ))

    fig.update_layout(
        title=dict(
            text=f"Firefox Bug Cluster Analysis by Priority (Bug-Level)<br>"
                 f"<sup>Hover for details | Core=●  Outlier=✕ | "
                 f"Silhouette={sil_score:.3f} | Davies-Bouldin={db_score:.3f}</sup>",
            font=dict(size=16)
        ),
        xaxis_title="UMAP Dimension 1",
        yaxis_title="UMAP Dimension 2",
        legend=dict(title="Priority", font=dict(size=11), bordercolor="#CCCCCC", borderwidth=1),
        plot_bgcolor="#F8F9FA", paper_bgcolor="#FFFFFF",
        width=1200, height=800, hovermode="closest"
    )
    fig.write_html(OUTPUT_HTML)
    print(f"  Saved: {OUTPUT_HTML}")

    # ── OPTION 2: Static Matplotlib ───────────────────────────────────────────
    print(f"\nStep 7b: Creating static matplotlib plot...")
    fig2, ax = plt.subplots(figsize=(18, 13))
    ax.set_facecolor("#F8F9FA")
    fig2.patch.set_facecolor("#FFFFFF")

    # Confidence ellipse on core points only
    for pri, data in pri_data.items():
        mask = pri_outlier_mask.get(pri, np.ones(len(data['x']), dtype=bool))
        core_x = np.array([data['x'][i] for i in range(len(data['x'])) if mask[i]])
        core_y = np.array([data['y'][i] for i in range(len(data['y'])) if mask[i]])
        if len(core_x) >= 3:
            draw_confidence_ellipse(ax, core_x, core_y, COLORS[pri], n_std=ELLIPSE_STD)

    # Core points
    for pri, data in pri_data.items():
        if not data["x"]:
            continue
        mask = pri_outlier_mask.get(pri, np.ones(len(data['x']), dtype=bool))
        core_x = [data['x'][i] for i in range(len(data['x'])) if mask[i]]
        core_y = [data['y'][i] for i in range(len(data['y'])) if mask[i]]
        core_counts = [data['label_counts'][i] for i in range(len(data['label_counts'])) if mask[i]]
        out_x = [data['x'][i] for i in range(len(data['x'])) if not mask[i]]
        out_y = [data['y'][i] for i in range(len(data['y'])) if not mask[i]]
        out_ids = [data['bug_ids'][i] for i in range(len(data['bug_ids'])) if not mask[i]]
        out_counts = [data['label_counts'][i] for i in range(len(data['label_counts'])) if not mask[i]]

        if core_x:
            sizes = [max(30, min(150, c * 5)) for c in core_counts]
            ax.scatter(core_x, core_y, c=COLORS[pri],
                       label=f"{pri} core (n={len(core_x)})",
                       alpha=0.75, s=sizes, marker='o',
                       edgecolors="white", linewidths=0.5)

        if out_x:
            out_sizes = [max(80, min(200, c * 8)) for c in out_counts]
            ax.scatter(out_x, out_y, c=COLORS[pri],
                       label=f"{pri} outliers (n={len(out_x)})",
                       alpha=0.9, s=out_sizes, marker='X',
                       edgecolors=COLORS[pri], linewidths=1.5)
            offsets = [(15, 10), (-15, 10), (15, -10), (-15, -10), (20, 0), (-20, 0), (0, 20), (0, -20), (25, 10),
                       (-25, 10), (10, 25), (10, -25)]
            for i in range(len(out_x)):
                ox, oy = offsets[i % len(offsets)]
                ax.annotate(
                    f"Bug {out_ids[i]}",
                    (out_x[i], out_y[i]),
                    fontsize=7, color=COLORS[pri], fontweight='bold',
                    xytext=(ox, oy), textcoords="offset points",
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              alpha=0.85, edgecolor=COLORS[pri], linewidth=1),
                    arrowprops=dict(arrowstyle='->', color=COLORS[pri], lw=0.8)
                )

        cx, cy = np.mean(data['x']), np.mean(data['y'])
        # Circle badge — white border then colored circle then text
        ax.plot(cx, cy, 'o', markersize=38, color='white', zorder=9)
        ax.plot(cx, cy, 'o', markersize=34, color=COLORS[pri], zorder=10, alpha=0.95)
        ax.text(cx, cy, pri,
                fontsize=12, fontweight="bold",
                color="white", ha="center", va="center",
                zorder=11, fontfamily='monospace')

    ax.set_title(
        f"Firefox Bug Cluster Analysis by Priority (Bug-Level)\n"
        f"Core ● | Outliers ✕ | Silhouette={sil_score:.3f} | "
        f"Davies-Bouldin={db_score:.3f} | ARI={ari_score:.3f} | NMI={nmi_score:.3f}",
        fontsize=12, fontweight="bold", pad=20
    )
    ax.set_xlabel("UMAP Dimension 1", fontsize=12)
    ax.set_ylabel("UMAP Dimension 2", fontsize=12)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9, edgecolor="#CCCCCC", ncol=2)
    ax.grid(True, alpha=0.2, linestyle="--")
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight")
    print(f"  Saved: {OUTPUT_PNG}")

    print(f"\n{'=' * 60}")
    print(f"  SUMMARY")
    print(f"  Total bugs:                {len(valid_bugs)}")
    print(f"  Silhouette Score:          {sil_score:.4f}")
    print(f"  Davies-Bouldin Index:      {db_score:.4f}")
    print(f"  Adjusted Rand Index:       {ari_score:.4f}")
    print(f"  Normalized Mutual Info:    {nmi_score:.4f}")
    print(f"  Homogeneity:               {h_score:.4f}")
    print(f"  Completeness:              {c_score:.4f}")
    print(f"  V-Measure:                 {v_score:.4f}")
    print(f"  Interactive:               {OUTPUT_HTML}")
    print(f"  Static:                    {OUTPUT_PNG}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run()
