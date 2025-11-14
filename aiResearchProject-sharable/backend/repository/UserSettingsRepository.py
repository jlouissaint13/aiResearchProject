import sqlite3
import bcrypt
class UserSettingsRepository:
    def __init__(self):
        pass
    #this function will update user info if they change the password
    def update_user_info_by_user_id_pw(self,user_id,email,username,password):
        con = sqlite3.connect("sql.db")
        cur = con.cursor()
        password_bytes = password.encode("utf-8")
        
        salt = bcrypt.gensalt()
        
        pw_hash = bcrypt.hashpw(password_bytes,salt)
        
        hashed_string = pw_hash.decode("utf-8")
        
        cur.execute("UPDATE USERS SET email = ? , username = ? , password = ? WHERE user_id = ? ",(email,username,hashed_string,user_id))
        con.commit()
        con.close()
    
    
    #this function will update user info if they choose not to change the password
    def update_user_info_by_user_id_no_pw(self, user_id, email, username):
        con = sqlite3.connect("sql.db")
        cur = con.cursor()
        cur.execute("UPDATE USERS SET email = ?, username = ? WHERE user_id = ?", (email, username, user_id))
        
        con.commit()
        con.close()
    
    
    def get_user_info_for_settings_page_by_user_id(self,user_id):
        con = sqlite3.connect("sql.db")
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cur.execute("SELECT email, username FROM USERS WHERE user_id = ?", (user_id,))
        res = cur.fetchone()
        con.close()
        return res
    #different from the registration and login user_exists because we need to exclude the current
    #account holder from the search
    def user_exists(self, user_id,email,username):
        con = sqlite3.connect("sql.db")
        cur = con.cursor()

        cur.execute("SELECT 1 FROM USERS WHERE email = ? AND user_id != ?", (email, user_id))
        if cur.fetchone():
            con.close()
            return True

        cur.execute("SELECT 1 FROM USERS WHERE username = ? AND user_id != ?", (username, user_id))
        if cur.fetchone():
            con.close()
            return True

        con.close()
        return False
    
    def delete_user_data_by_user_id(self,user_id):
        con = sqlite3.connect('sql.db')
        cur = con.cursor()
        cur.execute("DELETE FROM USERS where user_id = ?",(user_id,))
        cur.execute("DELETE FROM CONVERSATIONS WHERE user_id = ?",(user_id,))
        cur.execute("DELETE FROM MESSAGES WHERE user_id = ?",(user_id,))
        cur.execute("Delete FROM PDFMANAGER WHERE user_id = ?",(user_id,))
        cur.execute("Delete From MODELSETTINGS WHERE user_id = ?",(user_id,))
        con.commit()
        con.close()