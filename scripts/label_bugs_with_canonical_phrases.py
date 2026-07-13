import os
import json
import pandas as pd
from langchain_ollama import OllamaLLM

# ---------------- CONFIG ----------------
BATCH_SIZE = 10
MAX_SUMMARY_CHARS = 300
MAX_KEYPHRASE_CHARS = 200

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

# ---------------- LOAD FILES ----------------
bugs_df = pd.read_csv(os.path.join(base, "severity_csv_data_with_blockers.csv"))
clusters_df = pd.read_csv(os.path.join(base, "clusters_with_improved_canonical_with_tfid_scores.csv"))

canonical_phrases = clusters_df["canonical_phrase"].tolist()
canonical_set = set(canonical_phrases)
canonical_phrases_text = ", ".join(canonical_phrases)

print(f"Loaded {len(bugs_df)} bugs and {len(canonical_phrases)} canonical phrases.")

# ---------------- LLM ----------------
llm = OllamaLLM(
    model="mistral:7b-instruct-q4_K_M",
    temperature=0,
    num_predict=40,
    stop=["\n\n"]
)

# ---------------- HELPERS ----------------
def extract_summary(conversation_str):
    try:
        data = json.loads(conversation_str)
        return data.get("summary", "")[:MAX_SUMMARY_CHARS]
    except Exception:
        return str(conversation_str)[:MAX_SUMMARY_CHARS]


def extract_keyphrases(tfidf_str):
    try:
        data = json.loads(tfidf_str)
        phrases = data.get("tfidf_keyphrases", [])
        return ", ".join(p["phrase"] for p in phrases[:3])[:MAX_KEYPHRASE_CHARS]
    except Exception:
        return str(tfidf_str)[:MAX_KEYPHRASE_CHARS]


def chunk(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i: i + size]


# ---------------- BATCH LABELING ----------------
def label_batch(batch_rows):
    """
    batch_rows: list of (index, bug_id, summary, keyphrases)
    Returns: dict of {index: canonical_phrase}
    """
    prompt = f"""Read each bug and pick the single most relevant canonical phrase.
Return EXACTLY one line per bug in this format:
<bug_index>|<canonical_phrase>

Allowed canonical phrases:
{canonical_phrases_text}

Bugs:
"""
    for idx, bug_id, summary, keyphrases in batch_rows:
        prompt += f"\n{idx}\nSummary: {summary}\nKeyphrases: {keyphrases}\n"

    try:
        response = llm.invoke(prompt)
    except Exception as e:
        print(f"  ⚠ Ollama error: {e}")
        return {}

    results = {}
    for line in response.splitlines():
        if "|" not in line:
            continue
        left, right = line.split("|", 1)
        left = left.strip()
        right = right.strip()
        if left.isdigit() and right in canonical_set:
            results[int(left)] = right

    return results


# ---------------- MAIN ----------------
labels = {}
batches = list(chunk(list(bugs_df.iterrows()), BATCH_SIZE))
print(f"Processing {len(batches)} batches of up to {BATCH_SIZE} bugs each...\n")

for batch_num, batch in enumerate(batches):
    batch_rows = []
    for idx, row in batch:
        summary = extract_summary(row["conversation"])
        keyphrases = extract_keyphrases(row["tfidf_scores"])
        batch_rows.append((idx, row["bug_id"], summary, keyphrases))

    print(f"Batch {batch_num + 1}/{len(batches)} — indices {batch_rows[0][0]}–{batch_rows[-1][0]}")
    batch_labels = label_batch(batch_rows)
    labels.update(batch_labels)
    print(f"  ✅ Got {len(batch_labels)}/{len(batch_rows)} labels")

# ---------------- SAVE ----------------
bugs_df["canonical_label"] = bugs_df.index.map(labels)

output_path = os.path.join(base, "bugs_labeled_with_canonical_phrases.csv")
bugs_df.to_csv(output_path, index=False)

print(f"\n🎉 Done! Labeled {bugs_df['canonical_label'].notna().sum()}/{len(bugs_df)} bugs.")
print(f"Saved → bugs_labeled_with_canonical_phrases.csv")

# ---------------- LABEL DISTRIBUTION ----------------
print("\n📊 Bugs per canonical label:")
print(bugs_df["canonical_label"].value_counts().to_string())
