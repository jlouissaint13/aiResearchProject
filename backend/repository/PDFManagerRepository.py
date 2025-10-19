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
    
    
    def retrieve_all_pdfs_by_user_id(self,user_id):
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        cur = con.cursor()

        cur.execute("SELECT pdf_id, pdf_name, file_path FROM PDFMANAGER WHERE user_id = ?", (user_id,))
        res = cur.fetchall()
        con.commit()
        con.close()
        transformed_messages = [dict(row) for row in res]
        print(transformed_messages)
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
        
        