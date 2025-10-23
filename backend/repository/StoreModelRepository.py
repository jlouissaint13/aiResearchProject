import sqlite3



class StoreModelRepository:
    def __init__(self):
        self.db_path = 'sql.db'
    
    
    def store_model(self,provider,key_name,api_key):
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        if self.api_key_exists(api_key):
            return


        cur.execute(
            "Insert into STOREDMODELS(provider,key_name,api_key) VALUES (?,?,?)",
            (provider,key_name,api_key)
        )
        
        con.commit()
        con.close()
        
        
    def api_key_exists(self,api_key):
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        
        cur.execute("select api_key from STOREDMODELS where api_key = ?",(api_key,))
        
        res = cur.fetchone()
        
        
        if res is None:
            return False
        
        return True


    def update_api_key(self, old_api_key, new_api_key):
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()

    
        cur.execute("SELECT api_key FROM STOREDMODELS WHERE api_key = ?", (old_api_key,))
        if cur.fetchone() is None:
            con.close()
            return False

        cur.execute("UPDATE STOREDMODELS SET api_key = ? WHERE api_key = ?", (new_api_key, old_api_key))
        con.commit()
        con.close()
        return True
        
        