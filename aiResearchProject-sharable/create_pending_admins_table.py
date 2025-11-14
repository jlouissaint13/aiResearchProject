import sqlite3

with sqlite3.connect("sql.db") as conn:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS PENDING_ADMINS (
            request_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            email TEXT NOT NULL,
            first_name TEXT,
            password TEXT NOT NULL,
            approved INTEGER DEFAULT 0
        );
    """)
    conn.commit()

print(" PENDING_ADMINS table created (or already exists).")
