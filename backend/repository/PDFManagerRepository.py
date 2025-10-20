import sqlite3

class PDFManagerRepository:
    def __init__(self):
        self.db_path = 'sql.db'

    def insert_pdf(self, pdf_name, user_id, hash_value,file_path):
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()

        cur.execute(
            "INSERT INTO PDFMANAGER (pdf_name, user_id, hash_value,file_path) VALUES (?, ?, ?,?)",
            (pdf_name, user_id, hash_value,file_path)
        )

        con.commit()
        con.close()

    def select_by_user_id(self, user_id):
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()

        cur.execute(
            "SELECT pdf_id, pdf_name, user_id, hash_value FROM PDFMANAGER WHERE user_id = ?",
            (user_id,)
        )
        rows = cur.fetchall()

        con.close()
        return rows
    
    def retrieve_hash_value_by_file_path(self,file_path):
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        
        
        cur.execute("SELECT hash_value from PDFMANAGER where file_path= ?",(file_path,))
        res = cur.fetchone()
        con.close()
        return res
    
    
    
    def retrieve_all_pdfs_by_user_id(self,user_id):
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        cur = con.cursor()

        cur.execute("SELECT pdf_id, pdf_name, file_path FROM PDFMANAGER WHERE user_id = ?", (user_id,))
        res = cur.fetchall()
        con.commit()
        con.close()
        transformed_messages = [dict(row) for row in res]
        return transformed_messages

    def delete_by_user_id(self, user_id):
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()

        cur.execute(
            "DELETE FROM PDFMANAGER WHERE user_id = ?",
            (user_id,)
        )

        con.commit()
        con.close()
        
    def delete_pdf_by_user_id_and_hash_value(self,user_id,hash_value):
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        
        cur.execute("DELETE FROM PDFMANAGER WHERE user_id = ? AND hash_value = ?",(user_id,hash_value))
        
        con.commit()
        con.close()
     
    def get_ref_count(self,hash_value):
         con = sqlite3.connect(self.db_path)
         cur = con.cursor()
    
         cur.execute("SELECT COUNT(*) FROM PDFMANAGER WHERE hash_value = ? ",(hash_value,))
        
         #unpacks the tuple thats why you did this
         (res,) = cur.fetchone() 
         
         con.close()
         print(res)
         print(int(res))
         return res    
     
    def get_searchable_documents(self,user_id):
         con = sqlite3.connect(self.db_path)
         cur = con.cursor()
         
         cur.execute("SELECT hash_value FROM PDFMANAGER WHERE user_id = ?",(user_id,))
        
         result = cur.fetchall()

         hash_list = [row[0] for row in result]
        
         return hash_list
            
        
    def file_exists(self,user_id,hash_value):
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        
        cur.execute("SELECT hash_value FROM PDFMANAGER where user_id = ? and hash_value = ?",(user_id,hash_value,))
        
        res = cur.fetchone()
        
        if  res is not None: 
            return True
        
        return False
        