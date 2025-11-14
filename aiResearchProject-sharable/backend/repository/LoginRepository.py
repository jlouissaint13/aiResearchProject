import sqlite3
import bcrypt


class LoginRepository:

    def user_exists(self,username):
        con = sqlite3.connect("sql.db")
        cur = con.cursor()
        res = cur.execute("""
                      SELECT username FROM USERS
                      WHERE email = ? OR username = ?
                      """, (username,username))

        if res.fetchone() is not None:
            return True

        con.close()
        return False


    def check_pw(self,username,password):
        con = sqlite3.connect("sql.db")
        cur = con.cursor()
        res = cur.execute("""
                          SELECT password FROM USERS
                          WHERE email = ? OR username = ?
                          """, (username,username))
        stored_password = res.fetchone()[0]
        stored_password_bytes = stored_password.encode('utf-8')
        con.close()
        return bcrypt.checkpw(password.encode("utf-8"),stored_password_bytes)

    def get_id(self,username):
        con = sqlite3.connect("sql.db")
        cur = con.cursor()
        res = cur.execute("""
        SELECT user_id from USERS where email = ? OR username = ?
        """,(username,username))
        user_id = res.fetchone()[0]
        con.close()
        return user_id

    def get_name(self,username):
        con = sqlite3.connect("sql.db")
        cur = con.cursor()
        res = cur.execute("""
                      SELECT first_name from USERS where email = ? OR username = ?
                      """,(username,username))
        first_name = res.fetchone()[0]
        con.close()
        return first_name
    
    def is_active(self, username):
      with sqlite3.connect("sql.db") as conn:
        cur = conn.execute("SELECT is_active FROM USERS WHERE username=?", (username,))
        row = cur.fetchone()
        return bool(row and row[0] == 1)

