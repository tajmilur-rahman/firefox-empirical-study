import psycopg2
from psycopg2.extras import execute_batch
from langchain_ollama import OllamaLLM
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ---------------- CONFIG ----------------
BATCH_SIZE = 10  # Keep your original batch size
COMMIT_EVERY = 100
MAX_COMMENT_CHARS = 200  # Keep your original limit
MAX_WORKERS = 4  # Adjust based on your CPU

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
    password="mahalakshmi@07",
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

print(f"🚀 Found {total} uncategorized bugs")


# ---------------- LLM FACTORY (one per thread) ----------------
def create_llm():
    return OllamaLLM(
        model="mistral:7b-instruct-q4_K_M",
        temperature=0,
        num_predict=40,  # YOUR ORIGINAL SETTING
        stop=["\n\n"]  # YOUR ORIGINAL SETTING
    )


# ---------------- HELPERS ----------------
def chunk(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def trim(text):
    return text[:MAX_COMMENT_CHARS] if text else ""


# ---------------- BATCH PROCESSOR (YOUR EXACT LOGIC) ----------------
def process_batch(batch):
    """Process one batch - each thread gets its own LLM"""
    llm = create_llm()
    batch_updates = []

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
        print("⚠ Ollama error:", e)
        return []

    # YOUR EXACT PARSING LOGIC
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

        batch_updates.append((category, int(left)))

    return batch_updates


# ---------------- PARALLEL MAIN LOOP ----------------
processed = 0
updates = []
updates_lock = threading.Lock()

# Convert to list for parallel processing
batches = list(chunk(bugs, BATCH_SIZE))
print(f"📦 Processing {len(batches)} batches with {MAX_WORKERS} workers")

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    # Submit all batches
    future_to_batch = {executor.submit(process_batch, batch): i
                       for i, batch in enumerate(batches)}

    # Process results as they complete
    for future in as_completed(future_to_batch):
        batch_num = future_to_batch[future]

        try:
            batch_updates = future.result()

            # Thread-safe update
            with updates_lock:
                updates.extend(batch_updates)
                processed += len(batch_updates)

                # Commit in batches
                if len(updates) >= COMMIT_EVERY:
                    execute_batch(
                        cur,
                        "UPDATE bugs SET category_v2 = %s WHERE id = %s",
                        updates,
                        page_size=100
                    )
                    conn.commit()
                    updates.clear()
                    print(f"✅ Categorized {processed}/{total} (batch {batch_num + 1}/{len(batches)})")

        except Exception as e:
            print(f"⚠ Error processing batch {batch_num}: {e}")

# Final commit
if updates:
    with updates_lock:
        execute_batch(
            cur,
            "UPDATE bugs SET category_v2 = %s WHERE id = %s",
            updates,
            page_size=100
        )
        conn.commit()
        print(f"✅ Final commit: {len(updates)} updates")

print(f"🎉 DONE — categorized {processed} bugs")

cur.close()
conn.close()