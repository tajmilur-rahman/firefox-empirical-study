"""
MERGE SCRIPT – Merge Similar Named Clusters (via Ollama)
=========================================================
Reads the existing clusters_llm.csv and clusters_llm_canonical.csv,
asks the LLM to merge clusters with similar names, and saves updated files.

Run this AFTER script2_llm_clustering.py if you want cleaner clusters.
No need to re-run Script 2.

Prerequisites:
  pip install ollama pandas
  ollama pull mistral:7b-instruct-q4_K_M

Input:
  clusters_llm.csv           → flat cluster file from Script 2
  clusters_llm_canonical.csv → canonical phrases from Script 2

Output:
  clusters_llm_merged.csv           → updated flat cluster file
  clusters_llm_canonical_merged.csv → updated canonical phrases
"""

import json
import time
import re
import threading
import pandas as pd
import ollama

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_CLUSTERS   = "clusters_llm.csv"
INPUT_CANONICAL  = "clusters_llm_canonical.csv"
OUTPUT_CLUSTERS  = "llm_merged.csv"
OUTPUT_CANONICAL = "llm_canonical_merged.csv"
OLLAMA_MODEL     = "mistral:7b-instruct-q4_K_M"
MERGE_BATCH_SIZE = 50    # cluster names per LLM call
REQUEST_TIMEOUT  = 120   # seconds before giving up on a call
SLEEP_BETWEEN_CALLS = 1
# ──────────────────────────────────────────────────────────────────────────────


# ── Prompt ────────────────────────────────────────────────────────────────────
MERGE_SYSTEM_PROMPT = """You are a software bug analysis expert.
You are given a list of cluster names that were generated independently in batches.
Some of these names describe the same concept and should be merged.

Rules:
- Group cluster names that mean the SAME thing (e.g. "crash", "crash-related", "startup-crash" should be merged).
- Keep names that are clearly distinct in their own group.
- Choose ONE representative name for each merged group — the most specific/descriptive one.
- Every name in the input must appear in exactly one group in the output.
- Respond ONLY with a valid JSON object. No explanation, no markdown, no code fences.

Response format:
{
  "merges": {
    "chosen-canonical-name": ["old-name-1", "old-name-2", "old-name-3"],
    "another-canonical-name": ["old-name-4"]
  }
}
"""

MERGE_PROMPT_TEMPLATE = """Below is a list of cluster names. Group the ones that mean the same thing.

Cluster names:
{cluster_names_list}

Return ONLY the JSON object described in the system prompt."""
# ──────────────────────────────────────────────────────────────────────────────


def call_ollama_for_merge(name_batch: list[str], model: str) -> dict:
    """
    Send a batch of cluster names to Ollama and ask it to merge similar ones.
    Returns dict: { canonical_name: [old_name1, old_name2, ...] }
    Uses threading for timeout support on older ollama versions.
    """
    numbered  = "\n".join(f"{i+1}. {n}" for i, n in enumerate(name_batch))
    user_prompt = MERGE_PROMPT_TEMPLATE.format(cluster_names_list=numbered)

    result = {}
    error  = {}

    def run():
        try:
            response = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": MERGE_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                options={"temperature": 0.0, "num_predict": 1024},
            )
            result["response"] = response
        except Exception as e:
            error["msg"] = str(e)

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(timeout=REQUEST_TIMEOUT)

    if thread.is_alive():
        print(f"  [WARN] Call timed out after {REQUEST_TIMEOUT}s — keeping originals")
        return {}
    if "msg" in error:
        print(f"  [WARN] Call failed: {error['msg']} — keeping originals")
        return {}

    raw_text = result["response"]["message"]["content"].strip()
    raw_text = re.sub(r"```(?:json)?", "", raw_text).strip().rstrip("`").strip()

    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        print(f"  [WARN] No JSON found in response — keeping originals")
        return {}

    try:
        data = json.loads(match.group())
        return data.get("merges", {})
    except json.JSONDecodeError as e:
        print(f"  [WARN] JSON parse error: {e} — keeping originals")
        return {}


def build_merge_mapping(all_cluster_names: list[str], model: str) -> dict[str, str]:
    """
    Send cluster names in batches to the LLM and build a mapping:
      old_cluster_name → new_canonical_name

    Any name the LLM doesn't mention is kept as-is.
    """
    total = len(all_cluster_names)
    print(f"  {total} unique cluster names → sending in batches of {MERGE_BATCH_SIZE}\n")

    old_to_new: dict[str, str] = {}

    name_batches = [
        all_cluster_names[i:i+MERGE_BATCH_SIZE]
        for i in range(0, total, MERGE_BATCH_SIZE)
    ]

    for idx, name_batch in enumerate(name_batches):
        print(f"  Merge batch {idx+1}/{len(name_batches)}  ({len(name_batch)} names) …")
        merges = call_ollama_for_merge(name_batch, model)

        if merges:
            for canonical_name, old_names in merges.items():
                for old_name in old_names:
                    old_name = old_name.strip()
                    if old_name:
                        old_to_new[old_name] = canonical_name

        # Any names the LLM didn't mention → keep as-is
        for name in name_batch:
            if name not in old_to_new:
                old_to_new[name] = name

        time.sleep(SLEEP_BETWEEN_CALLS)

    return old_to_new


def apply_merge_to_clusters(clusters_df: pd.DataFrame,
                             canonical_df: pd.DataFrame,
                             old_to_new: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply the old→new name mapping to both DataFrames and reassign integer cluster IDs.

    clusters_df has: cluster_id, keyphrase, tf_score, bug_id, severity
    canonical_df has: cluster_id, canonical_phrase, phrase_count, phrases

    We join on cluster_id to get canonical_phrase into clusters_df,
    then apply the merge mapping and rebuild both files.
    """
    clusters_df  = clusters_df.copy()
    canonical_df = canonical_df.copy()

    # Step 1 — Join canonical_phrase into the flat clusters file via cluster_id
    id_to_name = canonical_df.set_index("cluster_id")["canonical_phrase"].to_dict()
    clusters_df["canonical_phrase"] = clusters_df["cluster_id"].map(id_to_name)

    # Step 2 — Apply the merge mapping (old name → new canonical name)
    clusters_df["canonical_phrase"] = clusters_df["canonical_phrase"].map(
        lambda name: old_to_new.get(name, name) if pd.notna(name) else name
    )

    # Step 3 — Re-assign integer cluster IDs based on merged names
    unique_names = sorted(clusters_df["canonical_phrase"].dropna().unique())
    name_to_id   = {name: i for i, name in enumerate(unique_names)}
    clusters_df["cluster_id"] = clusters_df["canonical_phrase"].map(name_to_id)

    # Step 4 — Rebuild canonical DataFrame — one row per merged cluster
    canonical_rows = []
    for cid_name, group in clusters_df.groupby("canonical_phrase"):
        all_phrases = group["keyphrase"].tolist()
        canonical_rows.append({
            "cluster_id":       name_to_id[cid_name],
            "canonical_phrase": cid_name,
            "phrase_count":     len(all_phrases),
            "phrases":          json.dumps(all_phrases),
        })

    new_canonical_df = pd.DataFrame(canonical_rows).sort_values("cluster_id")

    return clusters_df, new_canonical_df


def main():
    # ── Load existing outputs from Script 2 ──────────────────────────────────
    print(f"Loading {INPUT_CLUSTERS} …")
    clusters_df  = pd.read_csv(INPUT_CLUSTERS)
    print(f"Loading {INPUT_CANONICAL} …")
    canonical_df = pd.read_csv(INPUT_CANONICAL)

    before_count = clusters_df["cluster_id"].nunique()
    print(f"\n  Clusters before merge: {before_count}\n")

    # ── Build merge mapping ───────────────────────────────────────────────────
    print("Asking LLM to identify similar cluster names …\n")
    all_names = sorted(canonical_df["canonical_phrase"].unique().tolist())
    old_to_new = build_merge_mapping(all_names, OLLAMA_MODEL)

    # ── Apply merge ───────────────────────────────────────────────────────────
    clusters_df, canonical_df = apply_merge_to_clusters(clusters_df, canonical_df, old_to_new)

    after_count = clusters_df["cluster_id"].nunique()
    print(f"\n  Clusters before merge: {before_count}")
    print(f"  Clusters after merge:  {after_count}")
    print(f"  Reduced by:            {before_count - after_count} clusters\n")

    # ── Save ─────────────────────────────────────────────────────────────────
    clusters_df[["cluster_id", "keyphrase", "tf_score", "bug_id", "severity", "canonical_phrase"]]\
        .sort_values(["cluster_id", "tf_score"], ascending=[True, False])\
        .to_csv(OUTPUT_CLUSTERS, index=False)

    canonical_df.to_csv(OUTPUT_CANONICAL, index=False)

    print(f"Saved → {OUTPUT_CLUSTERS}")
    print(f"Saved → {OUTPUT_CANONICAL}\n")

    # ── Print sample of 5 largest merged clusters ─────────────────────────────
    print("── Sample: 5 largest clusters after merge ───────────────────────")
    top5 = canonical_df.nlargest(5, "phrase_count")
    for _, row in top5.iterrows():
        phrases = json.loads(row["phrases"])
        print(f"\n  Cluster {row['cluster_id']}  →  canonical: \"{row['canonical_phrase']}\"  ({row['phrase_count']} phrases)")
        for p in phrases[:6]:   # show first 6 phrases only
            print(f"    • {p}")
        if len(phrases) > 6:
            print(f"    … and {len(phrases) - 6} more")


if __name__ == "__main__":
    main()