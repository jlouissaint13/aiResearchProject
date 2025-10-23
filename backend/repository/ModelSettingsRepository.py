import sqlite3


class ModelSettingsRepository:
    def __init__(self):
        self.db_path = "sql.db"
        
    
    
    def change_settings_by_user_id(self,active_model,prompt_type,user_id):
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        
        if self.user_exists(user_id):
            cur.execute(
                "UPDATE model_settings SET active_model = ?, prompt_type = ? WHERE user_id = ?",
                (active_model, prompt_type, user_id)
            )
            con.commit()
            con.close()


        cur.execute(
            "INSERT INTO model_settings (active_model, prompt_type, user_id) VALUES (?, ?, ?)",
            (active_model, prompt_type, user_id)
        )
        con.commit()
        con.close()
    
    
    def user_exists(self,user_id):
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        
        res = cur.execute("select user_id from model_settings where user_id = ?",(user_id,))
        
        res = res.fetchone()
        
        if res is None:
            return False
        
        return True
        
        
        
    def delete_user_by_id(self,user_id):
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        
        cur.execute("Delete from model_settings where user_id = ?",(user_id))
        
        con.commit()
        con.close()