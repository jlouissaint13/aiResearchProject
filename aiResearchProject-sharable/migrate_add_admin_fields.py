# migrate_add_admin_fields.py
import sqlite3

DB_PATH = "sql.db"

def migrate():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. Add role column if missing
    try:
        cursor.execute("""
            ALTER TABLE USERS
            ADD COLUMN role TEXT NOT NULL DEFAULT 'user'
            CHECK (role IN ('user', 'admin'));
        """)
        print(" Added 'role' column to USERS.")
    except sqlite3.OperationalError:
        print(" 'role' column already exists — skipping.")

    # 2. Add is_active column if missing
    try:
        cursor.execute("""
            ALTER TABLE USERS
            ADD COLUMN is_active INTEGER NOT NULL DEFAULT 1;
        """)
        print(" Added 'is_active' column to USERS.")
    except sqlite3.OperationalError:
        print("'is_active' column already exists — skipping.")

    # 3. Create FEATURE_FLAGS table for guest toggle
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS FEATURE_FLAGS (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
    """)
    cursor.execute("""
        INSERT OR IGNORE INTO FEATURE_FLAGS (key, value)
        VALUES ('guest_enabled', '1');
    """)
    print(" Ensured FEATURE_FLAGS table exists and guest_enabled flag set to 1.")

    conn.commit()
    conn.close()
    print("\n Migration complete! Database updated successfully.")


if __name__ == "__main__":
    migrate()
