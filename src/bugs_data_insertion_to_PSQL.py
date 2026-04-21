import json
import os
import psycopg2
from psycopg2.extras import execute_batch, Json
from psycopg2.extensions import register_adapter
import db_connection

# --- Config ---
json_file = "/Users/Mrudhula_1/PycharmProjects/PythonProject/bdata/bugs.json"
table_name = "bugs"
batch_size = 1000

# --- Register JSON adapters ---
register_adapter(dict, Json)
register_adapter(list, Json)

# --- Connect to database ---
conn = db_connection.connect("bugs", "Mrudhula", os.getenv("DB_PASSWORD"), 5432)
cur = conn.cursor()

# --- Get table columns ---
cur.execute("""
    SELECT column_name
    FROM information_schema.columns
    WHERE table_name = %s
    ORDER BY ordinal_position
""", (table_name,))
columns = [c[0] for c in cur.fetchall()]
print(f"Detected {len(columns)} columns.")

cols_str = ", ".join(columns)
placeholders = ", ".join(["%s"] * len(columns))

# Remove ON CONFLICT - we want to keep all rows, even duplicates
# But this will only work if you've removed the PRIMARY KEY constraint on 'id'
sql = f"INSERT INTO {table_name} ({cols_str}) VALUES ({placeholders})"

# --- Define column types ---
jsonb_cols = {
    "history", "comments", "blocks", "see_also", "creator_detail",
    "flags", "mentors", "groups", "mentors_detail", "assigned_to_detail",
    "duplicates"
}
bool_cols = {"is_open", "is_confirmed", "is_creator_accessible"}
int_cols = {"id", "votes"}


# --- Helper to clean and convert values ---
def clean_value(col, value):
    if value is None:
        return None

    # JSONB columns - FIXED
    if col in jsonb_cols:
        if isinstance(value, (dict, list)):
            # Return dict/list directly - adapter will convert
            return value
        elif isinstance(value, str):
            # If it's a string, try to parse it first
            try:
                parsed = json.loads(value)
                return parsed
            except:
                # If parsing fails, return empty dict/list
                return {} if value.strip().startswith('{') else []
        else:
            # For any other type, return empty structure
            return None

    # Boolean columns
    if col in bool_cols:
        if isinstance(value, str):
            return value.lower() in ('true', '1', 'yes', 't')
        return bool(value)

    # Integer columns
    if col in int_cols:
        try:
            return int(value)
        except:
            return None

    # Text/timestamp columns - remove null bytes
    if isinstance(value, str):
        return value.replace("\u0000", "")

    return value  # Return as-is instead of converting everything to string


# --- Process JSON file and insert in batches ---
batch = []
row_count = 0
error_count = 0

with open(json_file, "r", encoding="utf-8") as f:
    for line_num, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue

        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Line {line_num}: JSON decode error - {e}")
            error_count += 1
            continue

        try:
            row = [clean_value(col, obj.get(col)) for col in columns]
            batch.append(row)
            row_count += 1

            if len(batch) >= batch_size:
                try:
                    execute_batch(cur, sql, batch)
                    conn.commit()  # Commit after each batch
                    print(f"Inserted {row_count} rows...")
                    batch = []
                except Exception as e:
                    conn.rollback()  # CRITICAL: Rollback the failed transaction
                    print(f"Batch insert failed at row {row_count}: {e}")
                    print("Trying row-by-row insertion for this batch...")

                    # Try inserting rows one by one to identify problem rows
                    successful = 0
                    for idx, single_row in enumerate(batch):
                        try:
                            cur.execute(sql, single_row)
                            conn.commit()
                            successful += 1
                        except Exception as row_error:
                            conn.rollback()
                            print(f"  Row {row_count - len(batch) + idx + 1} failed: {row_error}")
                            error_count += 1

                    print(f"  Successfully inserted {successful}/{len(batch)} rows from failed batch")
                    batch = []
        except Exception as e:
            print(f"Line {line_num}: Error processing row - {e}")
            print(f"Problematic data: {json.dumps(obj, indent=2)[:200]}...")
            error_count += 1
            continue

# Insert any remaining rows
if batch:
    try:
        execute_batch(cur, sql, batch)
        conn.commit()
        print(f"Inserted final batch: {row_count} total rows")
    except Exception as e:
        conn.rollback()
        print(f"Final batch insert failed: {e}")
        print("Trying row-by-row insertion for final batch...")

        successful = 0
        for idx, single_row in enumerate(batch):
            try:
                cur.execute(sql, single_row)
                conn.commit()
                successful += 1
            except Exception as row_error:
                conn.rollback()
                print(f"  Row {row_count - len(batch) + idx + 1} failed: {row_error}")
                error_count += 1

        print(f"  Successfully inserted {successful}/{len(batch)} rows from final batch")

# Close connection
cur.close()
conn.close()

print(f"✔ Completed: {row_count} rows inserted, {error_count} errors")