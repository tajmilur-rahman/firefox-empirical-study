import psycopg2
# Connect to PostgreSQL
def connect(db_name, user_name, password, port, host="localhost"):
    con = None
    try:
        con = psycopg2.connect(
            dbname=db_name,
            user=user_name,
            password=password,
            host=host,
            port=port
        )
        print("Connected to PostgreSQL")
    except Exception as e:
        print("Connection failed:", e)
    return con

def save(db):
    bug_id = db.get("id")
    title = db.get("title")

    con = connect("bugs", "Mrudhula", "DB_PASSWORD", 5432)
    if con:
        cur = con.cursor()

        # Create table if not exists
        cur.execute("""
            CREATE TABLE IF NOT EXISTS bugs (
                id BIGINT PRIMARY KEY,
                title TEXT
            );
        """)

        # Insert the bug
        cur.execute(
            "INSERT INTO bugs (id, title) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING;",
            (bug_id, str(title))
        )

        con.commit()
        cur.close()
        con.close()
        print(f"🐞 Saved bug {bug_id}")