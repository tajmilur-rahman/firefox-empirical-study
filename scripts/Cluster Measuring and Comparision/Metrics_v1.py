"""
Step 3: Compute cluster quality metrics for severity groups.

Metrics
-------
1. Silhouette Score      (overall + per severity)
2. Davies-Bouldin Index  (overall)
3. Intra-Cluster Distance  (per severity)
4. Inter-Cluster Distance  (every severity pair)

Input:  embeddings.npy, bugs_meta.csv
Output: cluster_quality_results.csv  (summary table)
        inter_cluster_distances.csv  (pairwise distances)
        cluster_metrics_report.txt   (human-readable report)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_distances
from itertools import combinations

# ── CONFIG ───────────────────────────────────────────────────────────────────
EMBEDDINGS_FILE = "embeddings.npy"
META_FILE        = "bugs_meta.csv"

OUT_SUMMARY   = "cluster_quality_results.csv"
OUT_INTER     = "inter_cluster_distances.csv"
OUT_REPORT    = "cluster_metrics_report.txt"

SEVERITY_ORDER = ["Blocker", "S1", "S2", "S3", "S4"]
# ─────────────────────────────────────────────────────────────────────────────

embeddings = np.load(EMBEDDINGS_FILE)
meta       = pd.read_csv(META_FILE)
meta["severity"] = meta["severity"].astype(str).str.strip()

# Encode severity as integer labels for sklearn
unique_severities = meta["severity"].unique().tolist()
label_map = {s: i for i, s in enumerate(unique_severities)}
int_labels = meta["severity"].map(label_map).values


# ── 1. SILHOUETTE SCORE ──────────────────────────────────────────────────────
print("Computing Silhouette scores …")
overall_silhouette = silhouette_score(embeddings, int_labels, metric="cosine")
per_sample_silhouette = silhouette_samples(embeddings, int_labels, metric="cosine")

per_severity_silhouette = {}
for sev in unique_severities:
    mask = meta["severity"] == sev
    per_severity_silhouette[sev] = per_sample_silhouette[mask].mean()


# ── 2. DAVIES-BOULDIN INDEX ──────────────────────────────────────────────────
print("Computing Davies-Bouldin Index …")
dbi = davies_bouldin_score(embeddings, int_labels)


# ── 3. INTRA-CLUSTER DISTANCE ────────────────────────────────────────────────
print("Computing Intra-Cluster distances …")
intra_distances = {}
for sev in unique_severities:
    mask = meta["severity"].values == sev
    sev_embs = embeddings[mask]
    if len(sev_embs) < 2:
        intra_distances[sev] = np.nan
        continue
    dist_matrix = cosine_distances(sev_embs)
    # Exclude diagonal (self-distance = 0)
    no_diag = dist_matrix[~np.eye(len(sev_embs), dtype=bool)]
    intra_distances[sev] = no_diag.mean()


# ── 4. INTER-CLUSTER DISTANCE ────────────────────────────────────────────────
print("Computing Inter-Cluster distances …")
inter_distances = {}
for sev_a, sev_b in combinations(unique_severities, 2):
    embs_a = embeddings[meta["severity"].values == sev_a]
    embs_b = embeddings[meta["severity"].values == sev_b]
    dist_matrix = cosine_distances(embs_a, embs_b)
    inter_distances[(sev_a, sev_b)] = dist_matrix.mean()


# ── BUILD SUMMARY TABLE ──────────────────────────────────────────────────────
rows = []
for sev in SEVERITY_ORDER:
    if sev not in unique_severities:
        continue
    count = (meta["severity"] == sev).sum()
    rows.append({
        "Severity":               sev,
        "Bug Count":              int(count),
        "Silhouette Score":       round(per_severity_silhouette.get(sev, np.nan), 4),
        "Intra-Cluster Distance": round(intra_distances.get(sev, np.nan), 4),
    })

summary_df = pd.DataFrame(rows)
summary_df.to_csv(OUT_SUMMARY, index=False)

# Inter-cluster table
inter_rows = [
    {"Severity A": a, "Severity B": b, "Inter-Cluster Distance": round(d, 4)}
    for (a, b), d in inter_distances.items()
]
inter_df = pd.DataFrame(inter_rows)
inter_df.to_csv(OUT_INTER, index=False)


# ── HUMAN-READABLE REPORT ────────────────────────────────────────────────────
report_lines = []
report_lines.append("=" * 60)
report_lines.append("CLUSTER QUALITY REPORT — Firefox Bug Severity Study")
report_lines.append("=" * 60)

report_lines.append(f"\n[Overall Silhouette Score]  {overall_silhouette:.4f}")
report_lines.append("  Range: -1 (bad) → +1 (perfect). >0.5 is good.")

report_lines.append(f"\n[Overall Davies-Bouldin Index]  {dbi:.4f}")
report_lines.append("  Lower is better. <1.0 is generally acceptable.")

report_lines.append("\n[Per-Severity Silhouette Scores]")
report_lines.append(f"  {'Severity':<10} {'Silhouette':>12}")
report_lines.append("  " + "-" * 24)
for sev in SEVERITY_ORDER:
    if sev in per_severity_silhouette:
        val = per_severity_silhouette[sev]
        report_lines.append(f"  {sev:<10} {val:>12.4f}")

report_lines.append("\n[Intra-Cluster Distances]  (lower = more cohesive)")
report_lines.append(f"  {'Severity':<10} {'Bugs':>6} {'Intra Dist':>12}")
report_lines.append("  " + "-" * 30)
for _, row in summary_df.iterrows():
    report_lines.append(f"  {row['Severity']:<10} {row['Bug Count']:>6} {row['Intra-Cluster Distance']:>12.4f}")

report_lines.append("\n[Inter-Cluster Distances]  (higher = better separated)")
report_lines.append(f"  {'Pair':<18} {'Inter Dist':>12}")
report_lines.append("  " + "-" * 32)
for _, row in inter_df.iterrows():
    pair = f"{row['Severity A']}-{row['Severity B']}"
    report_lines.append(f"  {pair:<18} {row['Inter-Cluster Distance']:>12.4f}")

report_lines.append("\n" + "=" * 60)
report_text = "\n".join(report_lines)

with open(OUT_REPORT, "w") as f:
    f.write(report_text)

print(report_text)
print(f"\n✅ Saved: {OUT_SUMMARY}, {OUT_INTER}, {OUT_REPORT}")