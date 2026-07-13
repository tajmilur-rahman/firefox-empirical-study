"""
SCRIPT 1 – Sentence Transformer Clustering
=========================================
Reads keyphrases from the dataset, generates embeddings using sentence-transformers,
computes cosine similarity, and clusters phrases using Agglomerative Clustering.
Also generates a canonical phrase for each cluster by selecting the phrase
closest to the cluster centroid (most representative phrase).

Output 1: priority_st.csv         → cluster_id, keyphrase, tf_score, bug_id, severity
Output 2: priority_st_canonical.csv → cluster_id, canonical_phrase, phrases, phrase_count
"""

import ast
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering
import json

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_CSV        = "/Users/Mrudhula/PycharmProjects/PythonProject/data/priority_125_with_tfidf.csv"
OUTPUT_CSV       = "priority_st.csv"
OUTPUT_CANONICAL = "priority_st_canonical.csv"
MODEL_NAME       = "all-MiniLM-L6-v2"   # fast, accurate for short phrases
# Distance threshold: lower → more clusters, higher → fewer/bigger clusters
# 0.35 works well for short technical keyphrases (tweak if needed)
DISTANCE_THRESHOLD = 0.35
# ──────────────────────────────────────────────────────────────────────────────


def load_keyphrases(csv_path: str) -> pd.DataFrame:
    """
    Load the CSV and explode tfidf_scores into individual rows.
    Returns columns:
      bug_id, priority, keyphrase, tf_score
    """

    df = pd.read_csv(
        csv_path,
        usecols=["bug_id", "priority", "tfidf_scores"]
    )

    rows = []

    for _, row in df.iterrows():
        try:
            tfidf_data = json.loads(row["tfidf_scores"])

            for item in tfidf_data["tfidf_keyphrases"]:
                rows.append({
                    "bug_id": row["bug_id"],
                    "priority": row["priority"],
                    "keyphrase": item["phrase"].strip(),
                    "tf_score": float(item["score"])
                })

        except Exception as e:
            print(f"Skipping bug {row['bug_id']}: {e}")

    return pd.DataFrame(rows)


def generate_embeddings(phrases: list[str], model_name: str) -> np.ndarray:
    """
    Use a SentenceTransformer model to encode all phrases into dense vectors.
    Vectors are L2-normalised so cosine similarity == dot product.
    """
    print(f"[1/3] Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"[2/3] Encoding {len(phrases)} phrases …")
    embeddings = model.encode(phrases, batch_size=64, show_progress_bar=True)
    return normalize(embeddings, norm="l2")   # shape: (N, 384)


def cluster_embeddings(embeddings: np.ndarray, distance_threshold: float) -> np.ndarray:
    """
    Apply Agglomerative Clustering with average linkage.
    'distance_threshold' in cosine-distance space (1 – cosine_similarity).
    Phrases closer than this threshold are merged into the same cluster.
    Returns array of cluster labels (one per phrase).
    """
    print(f"[3/3] Clustering (distance_threshold={distance_threshold}) …")
    # We pass precomputed distance matrix derived from cosine similarity
    similarity_matrix = embeddings @ embeddings.T            # cosine sim
    distance_matrix   = np.clip(1.0 - similarity_matrix, 0, 2)  # cosine distance

    model = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=distance_threshold,
    )
    labels = model.fit_predict(distance_matrix)
    print(f"    → {labels.max() + 1} clusters formed from {len(labels)} phrases")
    return labels


def generate_canonical_phrases(
    flat_df: pd.DataFrame,
    labels: np.ndarray,
    embeddings: np.ndarray,
    output_path: str,
) -> pd.DataFrame:
    """
    For each cluster, select the phrase whose embedding is closest to the
    cluster centroid — this is the most representative phrase in the cluster
    and becomes the canonical phrase.

    Strategy: centroid-closest (no external API needed)
      1. Compute the mean embedding of all phrases in the cluster (centroid).
      2. Pick the phrase with the highest cosine similarity to that centroid.
      3. This phrase best represents the shared meaning of the whole cluster.

    Saves clusters_st_canonical.csv with columns:
      cluster_id | canonical_phrase | phrases | phrase_count
    """
    flat_df = flat_df.copy()
    flat_df["cluster_id"] = labels

    canonical_rows = []

    for cid in sorted(flat_df["cluster_id"].unique()):
        # Get all row indices belonging to this cluster
        mask    = flat_df["cluster_id"] == cid
        indices = flat_df.index[mask].tolist()
        phrases = flat_df.loc[mask, "keyphrase"].tolist()

        # Get the embeddings for this cluster's phrases
        cluster_embeddings = embeddings[indices]   # shape: (k, 384)

        # Compute centroid = mean of all embeddings in the cluster
        centroid = cluster_embeddings.mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)   # re-normalise

        # Cosine similarity of each phrase to the centroid
        similarities = cluster_embeddings @ centroid     # shape: (k,)

        # The phrase with the highest similarity to centroid = canonical
        best_idx     = int(np.argmax(similarities))
        canonical    = phrases[best_idx]

        canonical_rows.append({
            "cluster_id":      cid,
            "canonical_phrase": canonical,
            "phrase_count":    len(phrases),
            "phrases":         json.dumps(phrases),   # full list as JSON string
        })

    canonical_df = pd.DataFrame(canonical_rows)
    canonical_df.to_csv(output_path, index=False)
    print(f"Saved → {output_path}  ({len(canonical_df)} canonical phrases)")

    # Print sample
    print("\n── Sample: canonical phrases for 5 largest clusters ────────────")
    top5 = canonical_df.nlargest(5, "phrase_count")
    for _, row in top5.iterrows():
        phrases = json.loads(row["phrases"])
        print(f"\n  Cluster {row['cluster_id']}  →  canonical: \"{row['canonical_phrase']}\"")
        for p in phrases:
            marker = "★" if p == row["canonical_phrase"] else "•"
            print(f"    {marker} {p}")

    return canonical_df


def save_clusters(flat_df: pd.DataFrame, labels: np.ndarray, output_path: str) -> pd.DataFrame:
    """
    Attach cluster labels to the flat DataFrame and save to CSV.
    Also prints a sample of the largest clusters for quick inspection.
    """
    flat_df = flat_df.copy()
    flat_df["cluster_id"] = labels

    # Reorder columns for readability
    result = flat_df[["cluster_id", "keyphrase", "tf_score", "bug_id", "priority"]]
    result = result.sort_values(["cluster_id", "tf_score"], ascending=[True, False])
    result.to_csv(output_path, index=False)
    print(f"Saved → {output_path}  ({len(result)} rows, {result['cluster_id'].nunique()} clusters)\n")

    return result


def main():
    # Step 1 – Load data
    print("Loading keyphrases …")
    flat_df = load_keyphrases(INPUT_CSV)
    print(f"  {len(flat_df)} keyphrases from {flat_df['bug_id'].nunique()} bugs\n")

    phrases = flat_df["keyphrase"].tolist()

    # Step 2 – Embeddings
    embeddings = generate_embeddings(phrases, MODEL_NAME)

    # Step 3 – Cluster
    labels = cluster_embeddings(embeddings, DISTANCE_THRESHOLD)

    # Step 4 – Save flat clusters (one row per phrase)
    save_clusters(flat_df, labels, OUTPUT_CSV)

    # Step 5 – Generate and save canonical phrase per cluster
    print("\nGenerating canonical phrases …")
    generate_canonical_phrases(flat_df, labels, embeddings, OUTPUT_CANONICAL)


if __name__ == "__main__":
    main()