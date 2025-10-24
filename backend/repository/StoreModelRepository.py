import sqlite3



class StoreModelRepository:
    def __init__(self):
        self.db_path = 'sql.db'
    
    
    def store_model(self,provider,key_name):
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        if self.api_key_exists(provider):
            cur.execute(
            "UPDATE STOREKEYNAME SET key_name = ? WHERE provider = ?",
            (key_name, provider)  
        )
        else:    
                 cur.execute(
            "Insert into STOREKEYNAME(provider,key_name) VALUES (?,?)",
            (provider,key_name)
        )

       
        
        con.commit()
        con.close()
        
        
    def api_key_exists(self,provider):
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        
        cur.execute("select provider from STOREKEYNAME where provider = ?",(provider,))
        
        res = cur.fetchone()
        
        con.close()
        if res is None:
            return False
        
        

        return True

