import sqlite3

from sympy.polys.polyconfig import query
from werkzeug.security import generate_password_hash

class UserRepository:
    def __init__(self):
        self.db = "sql.db"

    def save_reset_code(self, email, code, expires_at):
        with sqlite3.connect(self.db) as conn:
            
        
            
            
            conn.execute(
                "INSERT INTO password_resets (email,expires_at,reset_code) VALUES (?, ?, ?)",
                (email, expires_at, code)
            )
            conn.commit()

    def get_reset_code(self, email):
        query = "SELECT reset_code, expires_at FROM password_resets WHERE email=?"
        with sqlite3.connect(self.db) as conn:
            result = conn.execute(query, (email,)).fetchone()
            print(result)
            return result


    def get_by_email(self, email):
            with sqlite3.connect(self.db) as conn:
                cur = conn.execute("SELECT user_id, username, email FROM users WHERE email=?", (email,))
                row = cur.fetchone()
                if row:
                    return {"id": row[0], "username": row[1], "email": row[2]}
                return None



    def email_exists(self, email):
        with sqlite3.connect(self.db) as conn:
            cur = conn.execute("SELECT email FROM password_resets WHERE email = ?", (email,))
            row = cur.fetchone()
            return row[0] if row else None

    def update_password_by_email(self,email,new_password):
        with sqlite3.connect(self.db) as conn:
            conn.execute(
                "UPDATE USERS SET password = ? WHERE email = ?",
                (new_password, email)
            )
            conn.commit()

            return True


    def delete_reset_code(self,email):
        with sqlite3.connect(self.db) as conn:

            conn.execute("Delete from password_resets where email=?", (email,))
            conn.commit()        