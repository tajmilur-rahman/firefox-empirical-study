#!/usr/bin/env python3
"""
Bug Labeling using BERTScore - Priority Version (Improved)
Improvements:
1. Threshold raised to 0.85
2. All comments used (not just first) up to 500 chars
3. Single words kept (MIN_PHRASE_WORDS = 1)
4. Max 10 labels per bug (sorted by score)

Usage:
    python bug_labeler_bertscore_priority.py
"""

import sys
import csv
import json
import re
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("bert_score").setLevel(logging.ERROR)

csv.field_size_limit(10 * 1024 * 1024)

def install(pkg):
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

try:
    from bert_score import BERTScorer
except ImportError:
    install("bert-score")
    from bert_score import BERTScorer

try:
    import torch
except ImportError:
    install("torch")
    import torch

try:
    import numpy as np
except ImportError:
    install("numpy")
    import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
BUGS_CSV          = r"C:\Users\avish\Downloads\priority_125_with_tfidf.csv"
CLUSTERS_CSV      = r"C:\Users\avish\Downloads\priority_gemini_canonical.csv"
OUTPUT_CSV        = r"C:\Users\avish\Downloads\priority_bugs_labeled_bertscore.csv"
THRESHOLD         = 0.75   # Raised to 0.75 to reduce noise

MAX_LABELS        = 10     # Max labels per bug sorted by score
MIN_PHRASE_WORDS  = 1      # Keep single words — priority clusters use them
BERT_MODEL        = "distilbert-base-uncased"
BATCH_SIZE        = 128
# ─────────────────────────────────────────────────────────────────────────────

def extract_description(conversation_raw):
    """Use summary + first comment, 500 chars"""
    try:
        data = json.loads(conversation_raw)
        parts = []
        if "summary" in data:
            parts.append(data["summary"])
        comments = data.get("comments", [])
        if comments:
            parts.append(comments[0].get("text", ""))
        text = " ".join(parts)
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:500]
    except Exception:
        return str(conversation_raw)[:500]


def load_clusters(clusters_csv):
    clusters = []
    with open(clusters_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            phrases = json.loads(row["phrases"]) if row["phrases"] else []
            # Keep all phrases including single words
            filtered = [p for p in phrases if len(p.strip().split()) >= MIN_PHRASE_WORDS]
            if filtered:
                clusters.append({
                    "cluster_id":       row["cluster_id"],
                    "canonical_phrase": row["canonical_phrase"],
                    "phrases":          filtered,
                })
    print(f"  Loaded {len(clusters)} clusters")
    return clusters


def load_bugs(bugs_csv):
    bugs = []
    with open(bugs_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            description = extract_description(row.get("conversation", ""))
            bugs.append({
                "bug_id":      row["bug_id"],
                "priority":    row["priority"],
                "description": description
            })
    print(f"  Loaded {len(bugs)} bugs")
    return bugs


def run():
    print("Loading data...")
    clusters = load_clusters(CLUSTERS_CSV)
    bugs     = load_bugs(BUGS_CSV)

    print(f"\nSettings:")
    print(f"  Threshold:        {THRESHOLD}")
    print(f"  Description:      Summary + first comment (500 chars)")
    print(f"  Min phrase words: {MIN_PHRASE_WORDS}")
    print(f"  Max labels/bug:   {MAX_LABELS}")

    # Build flat list of all original keyphrases
    print("\nBuilding phrase index...")
    all_cluster_phrases = []
    cluster_phrase_map  = []
    cluster_indices_map = {}

    for c_idx, cluster in enumerate(clusters):
        cluster_indices_map[c_idx] = []
        for phrase in cluster["phrases"]:
            idx = len(all_cluster_phrases)
            all_cluster_phrases.append(phrase)
            cluster_phrase_map.append(c_idx)
            cluster_indices_map[c_idx].append(idx)

    cluster_phrase_map = np.array(cluster_phrase_map)

    print(f"  Total cluster phrases: {len(all_cluster_phrases)}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device.upper()}")

    # Load BERTScorer ONCE
    print(f"\nLoading BERTScorer ({BERT_MODEL})...")
    scorer = BERTScorer(
        model_type=BERT_MODEL,
        lang="en",
        rescale_with_baseline=False,
        device=device,
        batch_size=BATCH_SIZE
    )
    print("  BERTScorer loaded!\n")

    results = []
    n_bugs = len(bugs)

    print(f"Labeling {n_bugs} bugs...")

    for bug_idx, bug in enumerate(bugs):
        desc = bug["description"]
        if not desc.strip():
            desc = bug["bug_id"]

        # BERTScore: bug description vs ALL original cluster keyphrases
        candidates = [desc] * len(all_cluster_phrases)
        _, _, F1 = scorer.score(candidates, all_cluster_phrases)
        f1_scores = F1.numpy()

        # For each cluster get max BERTScore across its phrases
        cluster_max_scores  = np.zeros(len(clusters))
        cluster_best_phrase = [""] * len(clusters)

        for c_idx in range(len(clusters)):
            indices = cluster_indices_map[c_idx]
            if not indices:
                continue
            scores = f1_scores[indices]
            best_local = int(np.argmax(scores))
            cluster_max_scores[c_idx]  = scores[best_local]
            cluster_best_phrase[c_idx] = all_cluster_phrases[indices[best_local]]

        # Collect ALL clusters above threshold
        matched = []
        for c_idx in range(len(clusters)):
            if cluster_max_scores[c_idx] >= THRESHOLD:
                matched.append({
                    "label":  clusters[c_idx]["canonical_phrase"],
                    "phrase": cluster_best_phrase[c_idx],
                    "score":  cluster_max_scores[c_idx]
                })

        # Sort by score and keep top MAX_LABELS
        matched = sorted(matched, key=lambda x: -x["score"])[:MAX_LABELS]

        # If no match found → assign best available
        if not matched:
            best_c = int(np.argmax(cluster_max_scores))
            matched = [{
                "label":  clusters[best_c]["canonical_phrase"],
                "phrase": cluster_best_phrase[best_c],
                "score":  cluster_max_scores[best_c]
            }]

        matched_labels  = [m["label"] for m in matched]
        matched_phrases = [f"{m['phrase']}({m['score']:.3f})" for m in matched]

        print(f"  [{bug_idx+1}/{n_bugs}] {bug['bug_id']} ({bug['priority']}) → {len(matched_labels)} label(s): {matched_labels[:3]}")

        results.append({
            "bug_id":          bug["bug_id"],
            "priority":        bug["priority"],
            "matched_labels":  "; ".join(matched_labels),
            "matched_phrases": "; ".join(matched_phrases),
            "label_count":     len(matched_labels),
        })

    # Write output
    print(f"\nWriting to {OUTPUT_CSV}...")
    fieldnames = ["bug_id", "priority", "matched_labels", "matched_phrases", "label_count"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    multi = sum(1 for r in results if r["label_count"] > 1)
    avg   = round(sum(r["label_count"] for r in results) / len(results), 2)

    print(f"\n{'='*75}")
    print(f"  Done! {len(results)} bugs labeled")
    print(f"  Multiple labels: {multi} bugs had 2+ labels")
    print(f"  Avg labels/bug:  {avg}")
    print(f"\n  Sample results:")
    print(f"  {'Bug ID':<10} {'Pri':<6} {'Labels':<50} {'Count':>5}")
    print(f"  {'-'*75}")
    for r in results[:10]:
        print(f"  {r['bug_id']:<10} {r['priority']:<6} {r['matched_labels'][:48]:<50} {r['label_count']:>5}")
    print(f"{'='*75}")


if __name__ == "__main__":
    run()