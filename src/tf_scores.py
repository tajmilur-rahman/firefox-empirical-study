import pandas as pd
import json
import os
import glob
import psycopg2
from dotenv import load_dotenv

load_dotenv()

# Connect to PostgreSQL
conn = psycopg2.connect(
    dbname="bugbug",
    user="Mrudhula",
    password=os.getenv("DB_PASSWORD"),
    host="localhost",
    port="5432"
)
cur = conn.cursor()

# Build a lookup of bug_id -> conversation JSON from the individual files
conversation_lookup = {}
for filepath in glob.glob("/path/to/outputs/bug_*_conversation.json"):
    with open(filepath, 'r') as f:
        data = json.load(f)
    conversation_lookup[str(data['bug_id'])] = data

sheets = ['S1', 'S2', 'S3', 'S4']

for sheet_name in sheets:
    print(f"\nProcessing sheet: {sheet_name}")

    df = pd.read_excel(
        "/Users/Mrudhula/OneDrive - Gannon University/TF_IDF_score_report.xlsx",
        sheet_name=sheet_name
    )

    df.columns = df.columns.str.strip()
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Rows: {len(df)}")

    for _, row in df.iterrows():
        bug_id = str(int(row['Bug_id']))

        # Get full conversation from JSON file instead of truncated Excel cell
        if bug_id not in conversation_lookup:
            print(f"  WARNING: No conversation file found for bug_id {bug_id}, skipping.")
            continue

        conv_data = conversation_lookup[bug_id]
        conversation_str = json.dumps(conv_data.get('conversation'))

        # Get severity from Excel column, fallback to JSON file
        severity = row['Severity'] if pd.notna(row['Severity']) else conv_data.get('severity')

        if pd.isna(severity) or severity is None:
            print(f"  Skipping bug_id {bug_id} — severity is null")
            continue

        # Parse key_phrases JSON from Excel (much shorter, won't be truncated)
        key_phrases_json = json.loads(row['Key_phrases, TF-IDF score'])
        phrases = [p['phrase'] for p in key_phrases_json['phrases']]
        tf_scores = [p['tf'] for p in key_phrases_json['phrases']]
        tf_idf_scores = [p['tf_idf'] for p in key_phrases_json['phrases']]

        cur.execute("""
            INSERT INTO tf_scores_sample (bug_id, severity, conversation, key_phrases, tf_scores, tf_idf_scores)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (bug_id, severity, conversation_str, json.dumps(phrases), json.dumps(tf_scores), json.dumps(tf_idf_scores)))

    conn.commit()
    print(f"  Sheet '{sheet_name}' inserted successfully.")

cur.close()
conn.close()
print("\nAll sheets processed.")