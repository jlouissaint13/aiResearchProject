import sqlite3
from datetime import datetime



class MessageRepository:
    def __init__(self):
        pass


    def insert_message_by_id(self, message_id, conversation_id, content, role, user_id):
        con = sqlite3.connect("sql.db")
        cur = con.cursor()
        
        cur.execute("""
                    INSERT INTO MESSAGES (message_id,conversation_id,content,role,user_id,created_at)
                    values (?,?,?,?,?,?)
                    """,(message_id,conversation_id,content,role,user_id,datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        con.commit()
        con.close()
        
        
    def retrieve_all_messages_by_id(self,user_id,conversation_id):
        con = sqlite3.connect('sql.db')
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        
        cur.execute('select content,role,message_id from MESSAGES where conversation_id = ? and user_id = ? ORDER BY created_at ASC',(conversation_id,user_id))
        
        res = cur.fetchall()
        con.commit()
        con.close()
        transformed_messages = [dict(row) for row in res]
        return transformed_messages
        
    def delete_all_messages_by_id(self,conversation_id,user_id):
        con = sqlite3.connect("sql.db")
        cur = con.cursor()
        cur.execute("DELETE FROM MESSAGES WHERE conversation_id = ? and user_id = ?", (conversation_id,user_id))
        con.commit()
        con.close()