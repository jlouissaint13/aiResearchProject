import sqlite3


class ModelSettingsRepository:
    def __init__(self):
        self.db_path = "sql.db"



    def change_settings_by_user_id(self, active_model, prompt_type,provider,user_id):


        sql = """
        INSERT OR REPLACE INTO MODELSETTINGS 
            (active_model, prompt_type,current_provider,user_id) 
        VALUES 
            (?, ?, ?,?)
        """

        try:
            with sqlite3.connect(self.db_path) as con:
                con.execute(sql, (active_model, prompt_type,provider, user_id))
        except sqlite3.Error as e:
            print(e)
    
    def retrieve_user_settings_by_user_id(self,user_id):
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()


        cur.execute("Select active_model,prompt_type,current_provider from MODELSETTINGS where user_id = ?",(user_id,))

        res = cur.fetchone()
        con.close()
        if res is not None:
            return res
        else:
            return None

        
        
     #probably don't need this one
    def delete_settings_by_id(self,user_id):
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        
        cur.execute("Delete from model_settings where user_id = ?",(user_id))
        
        con.commit()
        con.close()