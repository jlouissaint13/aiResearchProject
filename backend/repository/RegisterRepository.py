import sqlite3
import uuid


class RegisterRepository:
    def create_account(self, first_name, email, username, password):
        user_id = str(uuid.uuid4())

        con = sqlite3.connect("sql.db")
        cur = con.cursor()

        cur.execute("""
                       INSERT INTO users (user_id,first_name, email, username, password)
                       VALUES (?,?, ?, ?, ?)
                       """, (user_id,first_name, email, username, password))
        con.commit()
        con.close()

    #if user does exist return true
    def user_exists(self, email, username):
        con = sqlite3.connect("sql.db")
        cur = con.cursor()
        res_email = cur.execute("Select email from USERS where email=?",
                               (email,))

        if res_email.fetchone() is not None:
            print("user exists email")
            return True

        res_username = cur.execute("Select username From Users where username=?",
                                  (username,))
        if res_username.fetchone() is not None:
            print("user exists")

            return True

        return False

