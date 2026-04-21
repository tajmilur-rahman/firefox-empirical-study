import psycopg2
import os
from langchain_ollama import OllamaLLM
from pathlib import Path

# ---------------- CONFIG ----------------
BATCH_SIZE = 30
COMMIT_EVERY = 100
MAX_COMMENT_CHARS = 200

CATEGORIES = {
    "Networking & Security": "Network issues, security vulnerabilities, auth, encryption, exploits",
    "Performance": "Slow behavior, freezes, memory leaks, high CPU or RAM usage",
    "UI / UX": "Layout, buttons, menus, accessibility, visual bugs",
    "Compatibility": "OS-specific, browser standards, platform-dependent behavior",
    "Privacy & Data": "User data handling, tracking, cookies, permissions",
    "Media & Extensions": "Audio/video playback, plugins, extensions",
    "Installation & Updates": "Install failures, upgrades, config persistence",
    "Developer Tools": "Debugger, console, devtools, JS engine",
    "File & System": "Downloads, uploads, filesystem access",
    "Session & Sync": "Login sessions, sync across devices"
}

CATEGORY_SET = set(CATEGORIES)
CATEGORY_TEXT = ", ".join(CATEGORIES)

# ---------------- DB ----------------
conn = psycopg2.connect(
    dbname="bugbug",
    user="Mrudhula",
    password=os.getenv("DB_PASSWORD"),
    host="localhost",
    port=5432
)
cur = conn.cursor()

# ---------------- LOAD SQL ----------------
sql_file = Path("../data/DbMigration.sql").read_text()

# Split SQL statements by semicolon
statements = [s.strip() for s in sql_file.split(";") if s.strip()]

# Execute the first statement (ALTER TABLE)
cur.execute(statements[0])
conn.commit()

# Execute the second statement (SELECT) and fetch results
cur.execute(statements[1])
bugs = cur.fetchall()
total = len(bugs)

print(f"Found {total} uncategorized bugs")

# ---------------- LLM ----------------
llm = OllamaLLM(
    model="mistral:7b-instruct-q4_K_M",
    temperature=0,
    num_predict=40,          # HARD CAP
    stop=["\n\n"]            # STOP rambling
)
# ---------------- HELPERS ----------------
def chunk(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

def trim(text):
    return text[:MAX_COMMENT_CHARS] if text else ""

# ---------------- MAIN LOOP ----------------
processed = 0
updates = []

for batch in chunk(bugs, BATCH_SIZE):
    prompt = f"""
    Read the context and decide the suitable categorization for each bug:
    Return only the category:
    Return EXACTLY one line:
    <bug_id>|<category>

    Allowed categories:
    {CATEGORY_TEXT}

Bugs:
"""

    for bug_id, summary, comments in batch:
        prompt += f"""
{bug_id}
Summary: {summary}
Comments: {trim(comments)}
"""

    try:
        response = llm.invoke(prompt)
    except Exception as e:
        print("Ollama error:", e)
        continue

    for line in response.splitlines():
        if "|" not in line:
            continue

        left, right = line.split("|", 1)
        left = left.strip().replace("ID", "").strip()

        if not left.isdigit():
            continue

        category = right.strip()
        if category not in CATEGORY_SET:
            continue

        updates.append((category, int(left)))
        processed += 1

    if len(updates) >= COMMIT_EVERY:
        cur.executemany(
            "UPDATE bugs SET category_v2 = %s WHERE id = %s",
            updates
        )
        conn.commit()
        updates.clear()
        print(f"Categorized {processed}/{total}")

# Final commit
if updates:
    cur.executemany(
        "UPDATE bugs SET category_v2 = %s WHERE id = %s",
        updates
    )
    conn.commit()

print(f"DONE — categorized {processed} bugs")

cur.close()
conn.close()
