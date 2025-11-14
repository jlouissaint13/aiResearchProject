import sqlite3
from datetime import datetime



class ConversationRepository:
    def __init__(self):
        pass


    def create_conversation(self, conversation_id, user_id, title):
        con = sqlite3.connect("sql.db")
        cur = con.cursor()

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cur.execute(
        'INSERT INTO CONVERSATIONS (conversation_id, user_id, created_at, title, modified_at) VALUES (?, ?, ?, ?, ?)',
        (conversation_id, user_id, current_time, title, current_time)
        )

        con.commit()
        con.close()


    def select_conversations_by_id(self, user_id):
        with sqlite3.connect("sql.db") as con:
            cur = con.cursor()
            cur.execute('SELECT * FROM CONVERSATIONS WHERE user_id = ? ORDER BY modified_at DESC', (user_id,))
            return cur.fetchall()
        
        
    def delete_conversation(self, conversation_id, user_id):
        con = sqlite3.connect("sql.db")
        cur = con.cursor()
        cur.execute("DELETE FROM CONVERSATIONS WHERE conversation_id = ? and user_id = ?", (conversation_id,user_id))
        con.commit()
        con.close()
        
    def conversation_last_modified_at(self,conversation_id,user_id):
        con = sqlite3.connect("sql.db")
        cur = con.cursor()
        cur.execute("UPDATE CONVERSATIONS SET modified_at =? WHERE conversation_id = ? and user_id = ?",
                    (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),conversation_id,user_id))
        con.commit()
        con.close()
        
    def retrieve_converstion_id_by_title_and_user_id(self,title,user_id):
        
            with sqlite3.connect("sql.db") as con:
                cur = con.cursor()
                cur.execute(
                "SELECT conversation_id FROM CONVERSATIONS WHERE normalized_title = ? AND user_id = ?",
                (title, user_id)
        )
                res = cur.fetchone()
            return res
      
    
    def conversation_exists(self,conversation_id,user_id):
        con = sqlite3.connect('sql.db')
        cur = con.cursor()

        res = cur.execute(
            'SELECT conversation_id FROM CONVERSATIONS WHERE conversation_id = ? AND user_id = ?',
            (conversation_id, user_id)
        )
        test = res.fetchone()
        print(test)
        if test is not None:
            return True
        
        return False
        