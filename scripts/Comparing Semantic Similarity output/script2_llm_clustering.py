"""
SCRIPT 2 – LLM-Based Semantic Grouping (Ollama / mistral:7b-instruct-q4_K_M)
==============================================================================
Sends batches of keyphrases to a local Ollama LLM and asks it to group
semantically similar phrases. Parses the JSON response and assigns cluster IDs.
The LLM-generated group name is used directly as the canonical phrase for each cluster.

Features:
  - Timeout per batch to prevent hanging
  - Automatic retry with smaller sub-batches if a batch fails
  - Singleton fallback if all retries are exhausted

Prerequisites:
  pip install ollama pandas
  ollama pull mistral:7b-instruct-q4_K_M

Output 1: clusters_llm.csv           → cluster_id, keyphrase, tf_score, bug_id, severity
Output 2: clusters_llm_canonical.csv → cluster_id, canonical_phrase, phrases, phrase_count
"""

import ast
import json
import re
import time
import pandas as pd
import ollama                  # pip install ollama
import threading

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_CSV             = "/Users/Mrudhula/PycharmProjects/PythonProject/data/keyphrases_severity_combined(in).csv"
OUTPUT_CSV            = "clusters_llm.csv"
OUTPUT_CANONICAL      = "clusters_llm_canonical.csv"
OLLAMA_MODEL          = "mistral:7b-instruct-q4_K_M"
BATCH_SIZE            = 20    # reduced from 40 — smaller = faster + more reliable
RETRY_BATCH_SIZE      = 10    # sub-batch size when retrying a failed batch
MAX_RETRIES           = 3     # max retry attempts per sub-batch
REQUEST_TIMEOUT       = 120   # seconds before a batch call is considered hung
SLEEP_BETWEEN_BATCHES = 1     # seconds to wait between batches
# ──────────────────────────────────────────────────────────────────────────────


# ── Prompt template ───────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a software bug analysis expert.
Your job is to group a list of bug-report keyphrases that describe the SAME underlying issue.

Rules:
- A phrase may belong to at most ONE group.
- Every phrase must appear in exactly one group.
- Use short, lowercase, hyphenated group names (e.g. "startup-crash", "login-failure").
- Respond ONLY with a valid JSON object. No explanation, no markdown, no code fences.

Response format:
{
  "groups": {
    "group-name-1": ["phrase A", "phrase B", "phrase C"],
    "group-name-2": ["phrase D"],
    ...
  }
}
"""

USER_PROMPT_TEMPLATE = """Group the following keyphrases by shared meaning.

Keyphrases:
{phrases_list}

Return ONLY the JSON object described in the system prompt."""
# ──────────────────────────────────────────────────────────────────────────────


def load_keyphrases(csv_path: str) -> pd.DataFrame:
    """Load CSV and explode keyphrase lists into individual rows."""
    df = pd.read_csv(csv_path, usecols=["bug_id", "severity", "key_phrases", "tf_scores"])
    rows = []
    for _, row in df.iterrows():
        try:
            phrases = ast.literal_eval(row["key_phrases"])
            scores  = ast.literal_eval(row["tf_scores"])
        except (ValueError, SyntaxError):
            continue
        for phrase, score in zip(phrases, scores):
            phrase = phrase.strip()
            if phrase:
                rows.append({
                    "bug_id":    row["bug_id"],
                    "severity":  row["severity"],
                    "keyphrase": phrase,
                    "tf_score":  float(score),
                })
    return pd.DataFrame(rows)

def call_ollama(phrases: list[str], model: str) -> dict:
    """
    Send a batch of phrases to the Ollama model and parse the JSON response.
    Uses threading to implement a manual timeout since older ollama versions
    don't support the timeout parameter.
    """
    numbered = "\n".join(f"{i+1}. {p}" for i, p in enumerate(phrases))
    user_msg = USER_PROMPT_TEMPLATE.format(phrases_list=numbered)

    result = {}
    error  = {}

    def run():
        try:
            response = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                options={
                    "temperature": 0.0,
                    "num_predict": 1024,
                },
            )
            result["response"] = response
        except Exception as e:
            error["msg"] = str(e)

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(timeout=REQUEST_TIMEOUT)   # wait max REQUEST_TIMEOUT seconds

    if thread.is_alive():
        print(f"  [WARN] Batch timed out after {REQUEST_TIMEOUT}s — will retry")
        return {}

    if "msg" in error:
        print(f"  [WARN] Batch failed: {error['msg']} — will retry")
        return {}

    raw_text = result["response"]["message"]["content"].strip()

    # Remove markdown code fences if present
    raw_text = re.sub(r"```(?:json)?", "", raw_text).strip().rstrip("`").strip()

    # Extract first {...} block
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        print(f"  [WARN] No JSON found in LLM response. Raw:\n{raw_text[:300]}")
        return {}

    try:
        data = json.loads(match.group())
        return data.get("groups", {})
    except json.JSONDecodeError as e:
        print(f"  [WARN] JSON parse error: {e}\n  Raw: {raw_text[:300]}")
        return {}

def process_batches(flat_df: pd.DataFrame, model: str, batch_size: int) -> pd.DataFrame:
    """
    Split all phrases into batches, call the LLM for each batch,
    and accumulate cluster assignments.

    If a batch fails:
      1. Split it into smaller sub-batches of RETRY_BATCH_SIZE
      2. Retry each sub-batch up to MAX_RETRIES times
      3. If all retries fail, assign each phrase its own singleton cluster

    The LLM returns group names like "startup-crash" or "login-failure".
    These group names become both the cluster label AND the canonical phrase.

    Returns flat_df with two new columns:
      - cluster_id     : integer ID for the cluster
      - canonical_phrase: the LLM-generated group name for that cluster
    """
    all_phrases = flat_df["keyphrase"].tolist()
    phrase_to_group: dict[str, str] = {}   # phrase → LLM group name

    batches = [all_phrases[i:i+batch_size] for i in range(0, len(all_phrases), batch_size)]
    print(f"Processing {len(all_phrases)} phrases in {len(batches)} batches …\n")

    for idx, batch in enumerate(batches):
        print(f"  Batch {idx+1}/{len(batches)}  ({len(batch)} phrases) …")
        groups = call_ollama(batch, model)

        if not groups:
            # ── Retry with smaller sub-batches ────────────────────────────
            print(f"  [RETRY] Splitting batch {idx+1} into sub-batches of {RETRY_BATCH_SIZE} …")
            sub_batches = [batch[i:i+RETRY_BATCH_SIZE] for i in range(0, len(batch), RETRY_BATCH_SIZE)]
            recovered = {}

            for sub_idx, sub_batch in enumerate(sub_batches):
                print(f"    Sub-batch {sub_idx+1}/{len(sub_batches)}  ({len(sub_batch)} phrases) …")
                success = False

                for attempt in range(1, MAX_RETRIES + 1):
                    sub_groups = call_ollama(sub_batch, model)
                    if sub_groups:
                        recovered.update(sub_groups)
                        print(f"    ✓ Sub-batch {sub_idx+1} succeeded on attempt {attempt}")
                        success = True
                        break
                    print(f"    [WARN] Sub-batch {sub_idx+1} attempt {attempt}/{MAX_RETRIES} failed — retrying …")
                    time.sleep(2)  # wait a bit longer before retrying

                if not success:
                    # All retries exhausted — give each phrase its own singleton
                    print(f"    [FALLBACK] Sub-batch {sub_idx+1} permanently failed — assigning singletons")
                    for phrase in sub_batch:
                        key = f"singleton-{phrase[:20].lower().replace(' ', '-')}"
                        recovered[key] = [phrase]

                time.sleep(SLEEP_BETWEEN_BATCHES)

            groups = recovered

        # ── Assign group names to phrases ─────────────────────────────────
        for group_name, group_phrases in groups.items():
            for phrase in group_phrases:
                phrase = phrase.strip()
                if phrase:
                    phrase_to_group[phrase] = group_name

        # Handle any phrases still missing after grouping
        for phrase in batch:
            if phrase not in phrase_to_group:
                phrase_to_group[phrase] = f"ungrouped-{phrase[:20].lower().replace(' ', '-')}"

        time.sleep(SLEEP_BETWEEN_BATCHES)

    # ── Convert group names → integer cluster IDs ────────────────────────────
    unique_names = sorted(set(phrase_to_group.values()))
    name_to_id   = {name: i for i, name in enumerate(unique_names)}

    flat_df = flat_df.copy()
    flat_df["cluster_id"] = flat_df["keyphrase"].map(
        lambda p: name_to_id.get(phrase_to_group.get(p, "ungrouped"), -1)
    )
    # Store the LLM group name alongside each phrase — used for canonical output
    flat_df["canonical_phrase"] = flat_df["keyphrase"].map(
        lambda p: phrase_to_group.get(p, "ungrouped")
    )
    return flat_df


def save_clusters(df: pd.DataFrame, output_path: str):
    """Save flat cluster file — one row per phrase."""
    result = df[["cluster_id", "keyphrase", "tf_score", "bug_id", "severity"]]
    result = result.sort_values(["cluster_id", "tf_score"], ascending=[True, False])
    result.to_csv(output_path, index=False)
    print(f"Saved → {output_path}  ({len(result)} rows, {result['cluster_id'].nunique()} clusters)\n")


def save_canonical(df: pd.DataFrame, output_path: str):
    """
    Build and save the canonical phrase file — one row per cluster.
    The canonical phrase is the LLM-generated group name (already in df).
    Since the LLM named the group, that name IS the canonical phrase —
    no extra API call needed.
    """
    canonical_rows = []

    for cid in sorted(df["cluster_id"].unique()):
        mask      = df["cluster_id"] == cid
        phrases   = df.loc[mask, "keyphrase"].tolist()
        canonical = df.loc[mask, "canonical_phrase"].iloc[0]

        canonical_rows.append({
            "cluster_id":       cid,
            "canonical_phrase": canonical,
            "phrase_count":     len(phrases),
            "phrases":          json.dumps(phrases),
        })

    canonical_df = pd.DataFrame(canonical_rows)
    canonical_df.to_csv(output_path, index=False)
    print(f"Saved → {output_path}  ({len(canonical_df)} canonical phrases)")

    # Print sample — 5 largest clusters
    print("\n── Sample: canonical phrases for 5 largest clusters ────────────")
    top5 = canonical_df.nlargest(5, "phrase_count")
    for _, row in top5.iterrows():
        phrases = json.loads(row["phrases"])
        print(f"\n  Cluster {row['cluster_id']}  →  canonical: \"{row['canonical_phrase']}\"")
        for p in phrases:
            print(f"    • {p}")


def main():
    print("Loading keyphrases …")
    flat_df = load_keyphrases(INPUT_CSV)
    print(f"  {len(flat_df)} keyphrases from {flat_df['bug_id'].nunique()} bugs\n")

    # Cluster phrases via LLM — also captures canonical group names
    flat_df = process_batches(flat_df, OLLAMA_MODEL, BATCH_SIZE)

    # Save flat cluster file
    save_clusters(flat_df, OUTPUT_CSV)

    # Save canonical phrase file
    print("Generating canonical phrase summary …")
    save_canonical(flat_df, OUTPUT_CANONICAL)


if __name__ == "__main__":
    main()