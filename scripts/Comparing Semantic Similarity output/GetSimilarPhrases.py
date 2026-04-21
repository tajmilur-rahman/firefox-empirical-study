import pandas as pd

# Load the CSV
df = pd.read_csv("clusters_llm_canonical.csv")

# Filter rows where phrase_count > 1
filtered = df[df["phrase_count"] > 1]

# Save the filtered result
filtered.to_csv("clusters_similar_phrases_llm.csv", index=False)

print(f"Saved {len(filtered)} clusters with more than one phrase.")