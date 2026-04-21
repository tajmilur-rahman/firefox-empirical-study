#!/usr/bin/env python3
"""
Keyphrase Semantic Similarity Grouper — PostgreSQL version
- unique_phrase = unique words from the phrase with HIGHER tf_score
  (higher tf_score = more important/relevant phrase)

Usage:
    python keyphrase_grouper_psql.py
"""

import sys
import json
import ast
import csv
import os
from collections import defaultdict

csv.field_size_limit(10 * 1024 * 1024)

def install(pkg):
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

try:
    import psycopg2
except ImportError:
    install("psycopg2-binary"); import psycopg2

try:
    import numpy as np
except ImportError:
    install("numpy"); import numpy as np

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    install("sentence-transformers")
    from sentence_transformers import SentenceTransformer, util

# ── Config ────────────────────────────────────────────────────────────────────
DB_HOST     = "localhost"
DB_PORT     = 5432
DB_NAME     = "bugbug"
DB_USER     = "Mrudhula"
DB_PASSWORD = os.getenv("DB_PASSWORD", "")  # replace with actual password

CSV_FILE             = "/Users/Mrudhula/Downloads/keyphrases_severity_combined(in).csv"
SIMILARITY_THRESHOLD = 0.75
TOP_N_PHRASES        = 5
BATCH_SIZE           = 256
# ─────────────────────────────────────────────────────────────────────────────

def get_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )


def parse_json_field(val):
    if not val:
        return []
    val = str(val).strip()
    try:
        return json.loads(val)
    except Exception:
        pass
    try:
        return ast.literal_eval(val)
    except Exception:
        return []


def load_category_map(cur):
    print("  Loading category_v3 from bugs table...")
    cur.execute("SELECT id, COALESCE(category_v3, 'Uncategorized') FROM bugs")
    cmap = {str(row[0]): row[1] for row in cur.fetchall()}
    print(f"  Loaded {len(cmap):,} bug categories")
    return cmap


def load_csv(category_map):
    print(f"  Reading CSV: {CSV_FILE}")
    records = []
    seen_phrases = set()

    with open(CSV_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bug_id   = str(row.get("bug_id", "")).strip()
            severity = str(row.get("severity", "Unknown")).strip()
            phrases  = parse_json_field(row.get("key_phrases", ""))
            scores   = parse_json_field(row.get("tf_scores", ""))
            category = category_map.get(bug_id, "Uncategorized")

            if not phrases:
                continue

            while len(scores) < len(phrases):
                scores.append(0.0)

            pairs = sorted(zip(phrases, scores), key=lambda x: -x[1])[:TOP_N_PHRASES]

            for phrase, score in pairs:
                phrase = str(phrase).strip()
                if not phrase:
                    continue
                dedup_key = (category, severity, phrase.lower())
                if dedup_key in seen_phrases:
                    continue
                seen_phrases.add(dedup_key)

                records.append({
                    "bug_id":   bug_id,
                    "severity": severity,
                    "category": category,
                    "phrase":   phrase,
                    "tf_score": float(score)
                })

    print(f"  Total unique phrases extracted: {len(records):,}")
    return records


def get_unique_phrase(phrase1, tf1, phrase2, tf2):
    """
    Extract unique words from the phrase with HIGHER tf_score.
    Higher tf_score = more important/relevant phrase in the bug context.
    Unique words = words in that phrase NOT present in the other phrase.
    Falls back to other phrase's unique words if primary has none.
    """
    set1 = set(phrase1.lower().split())
    set2 = set(phrase2.lower().split())

    unique1_words = [w for w in phrase1.split() if w.lower() not in set2]
    unique2_words = [w for w in phrase2.split() if w.lower() not in set1]

    unique1 = " ".join(unique1_words)
    unique2 = " ".join(unique2_words)

    if tf1 >= tf2:
        return unique1 if unique1 else unique2 if unique2 else "(no unique words)"
    else:
        return unique2 if unique2 else unique1 if unique1 else "(no unique words)"


def create_output_table(cur, conn):
    cur.execute("DROP TABLE IF EXISTS keyphrase_groups")
    conn.commit()

    cur.execute("""
        CREATE TABLE keyphrase_groups (
            pair_id          SERIAL PRIMARY KEY,
            category         TEXT,
            severity         TEXT,
            bug_id_1         TEXT,
            bug_id_2         TEXT,
            phrase_1         TEXT,
            phrase_2         TEXT,
            similarity_score FLOAT,
            common_phrase    TEXT,
            unique_phrase    TEXT,
            tf_score_1       FLOAT,
            tf_score_2       FLOAT
        )
    """)
    conn.commit()
    print("  keyphrase_groups table created.")


def run():
    print("Connecting to PostgreSQL...")
    conn = get_connection()
    conn.autocommit = False
    cur = conn.cursor()
    print(f"Connected to {DB_NAME} @ {DB_HOST}\n")

    # Step 1
    print("Step 1: Loading categories from PostgreSQL...")
    category_map = load_category_map(cur)

    # Step 2
    print("\nStep 2: Loading keyphrases CSV...")
    records = load_csv(category_map)
    if not records:
        print("No phrases found!")
        return

    # Step 3
    print("\nStep 3: Loading sentence-transformer model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("  Model loaded: all-MiniLM-L6-v2")

    # Step 4 — Find similar pairs
    print(f"\nStep 4: Finding similar pairs (threshold={SIMILARITY_THRESHOLD})...")
    grouped = defaultdict(list)
    for r in records:
        grouped[(r["category"], r["severity"])].append(r)

    print(f"  Found {len(grouped)} category+severity combinations")
    all_pairs = []

    for (category, severity), members in sorted(grouped.items()):
        phrases = [m["phrase"] for m in members]
        n = len(phrases)
        if n < 2:
            continue

        print(f"  [{category} | {severity}] {n} phrases → finding similar pairs...")

        embeddings = model.encode(
            phrases,
            batch_size=BATCH_SIZE,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        sim_matrix = util.cos_sim(embeddings, embeddings).numpy()

        for i in range(n):
            for j in range(i + 1, n):
                m1 = members[i]
                m2 = members[j]

                if m1["phrase"].lower().strip() == m2["phrase"].lower().strip():
                    continue

                sim = float(sim_matrix[i][j])
                if sim < SIMILARITY_THRESHOLD:
                    continue

                # Common phrase = closer to midpoint embedding
                midpoint = ((embeddings[i] + embeddings[j]) / 2).reshape(1, -1)
                sim_i = float(util.cos_sim(midpoint, embeddings[i].reshape(1, -1))[0][0])
                sim_j = float(util.cos_sim(midpoint, embeddings[j].reshape(1, -1))[0][0])
                common_phrase = m1["phrase"] if sim_i >= sim_j else m2["phrase"]

                # Unique phrase from higher tf_score phrase
                unique_phrase = get_unique_phrase(
                    m1["phrase"], m1["tf_score"],
                    m2["phrase"], m2["tf_score"]
                )

                all_pairs.append((
                    category,
                    severity,
                    m1["bug_id"],
                    m2["bug_id"],
                    m1["phrase"],
                    m2["phrase"],
                    round(sim, 4),
                    common_phrase,
                    unique_phrase,
                    m1["tf_score"],
                    m2["tf_score"]
                ))

    print(f"\n  Total similar pairs found: {len(all_pairs):,}")
    if not all_pairs:
        print(f"No pairs found — try lowering SIMILARITY_THRESHOLD")
        return

    # Step 5 — Insert
    print("\nStep 5: Creating table and inserting pairs...")
    create_output_table(cur, conn)

    insert_sql = """
        INSERT INTO keyphrase_groups
        (category, severity, bug_id_1, bug_id_2, phrase_1, phrase_2,
         similarity_score, common_phrase, unique_phrase, tf_score_1, tf_score_2)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    for i in range(0, len(all_pairs), 500):
        batch = all_pairs[i:i+500]
        cur.executemany(insert_sql, batch)
        conn.commit()
        print(f"  Inserted {min(i+500, len(all_pairs)):,}/{len(all_pairs):,} pairs...")

    # Step 6 — Summary
    cur.execute("SELECT COUNT(*) FROM keyphrase_groups")
    total = cur.fetchone()[0]

    cur.execute("""
        SELECT category, severity, phrase_1, phrase_2,
               similarity_score, common_phrase, unique_phrase,
               tf_score_1, tf_score_2
        FROM keyphrase_groups
        ORDER BY similarity_score DESC
        LIMIT 16
    """)
    top = cur.fetchall()

    print(f"\n{'='*100}")
    print(f"  Done! {total:,} pairs inserted into keyphrase_groups")
    print(f"\n  All pairs:")
    print(f"  {'Cat':<18} {'Sev':<6} {'Sim':>5} | {'Phrase 1':<25} | {'Phrase 2':<25} | {'Common':<20} | {'Unique':<20} | TF1  TF2")
    print(f"  {'-'*100}")
    for cat, sev, p1, p2, sim, common, unique, tf1, tf2 in top:
        print(f"  {(cat or '')[:16]:<18} {(sev or ''):<6} {sim:>5.3f} | {(p1 or '')[:23]:<25} | {(p2 or '')[:23]:<25} | {(common or '')[:18]:<20} | {(unique or '')[:18]:<20} | {tf1:.3f} {tf2:.3f}")
    print(f"{'='*100}")

    cur.close()
    conn.close()


if __name__ == "__main__":
    run()