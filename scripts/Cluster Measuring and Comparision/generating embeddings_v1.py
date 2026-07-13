import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# ── CONFIG ──────────────────────────────────────────────────────────────────
INPUT_CSV           = "/Users/Mrudhula/PycharmProjects/PythonProject/data/bugs_labeled.csv"   # <-- your file
LABELS_COLUMN       = "matched_labels"     # semicolon-separated canonical phrases
SEVERITY_COLUMN     = "severity"

OUTPUT_EMBEDDINGS   = "embeddings.npy"
OUTPUT_META         = "bugs_meta.csv"
# ────────────────────────────────────────────────────────────────────────────

df = pd.read_csv(INPUT_CSV)
df[SEVERITY_COLUMN] = df[SEVERITY_COLUMN].astype(str).str.strip()

# Drop rows with no labels
df = df.dropna(subset=[LABELS_COLUMN])
df = df[df[LABELS_COLUMN].str.strip() != ""]

print(f"Loaded {len(df)} bugs")
print(f"Severities found: {sorted(df[SEVERITY_COLUMN].unique())}")
print(f"Label count range: {df['label_count'].min()} – {df['label_count'].max()}")

# Replace semicolons with spaces so the model reads labels as a sentence
df["label_text"] = df[LABELS_COLUMN].str.replace(";", " ", regex=False)

model = SentenceTransformer("all-MiniLM-L6-v2")
print("\nEncoding embeddings …")
embeddings = model.encode(df["label_text"].tolist(), convert_to_numpy=True, show_progress_bar=True)

np.save(OUTPUT_EMBEDDINGS, embeddings)
df[["bug_id", SEVERITY_COLUMN]].reset_index(drop=True).to_csv(OUTPUT_META, index=False)

print(f"\n✅ Saved {OUTPUT_EMBEDDINGS}  shape={embeddings.shape}")
print(f"✅ Saved {OUTPUT_META}")