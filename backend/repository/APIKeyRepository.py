#If the user has an unsupported GPU they can use an api key that they pay for or go CPU mode which is likely very slow

#not being used right now but later I want the user to be able to choose by speed or accuracy which is why I have a list of models instead of just one
import sqlite3

conn = sqlite3.connect('../sql.db')
cursor = conn.cursor()
class APIKeyRepository:
    def __init__(self):
        pass

    def set_key(self):

        key = input("Please enter the API key: ")
        cursor.execute("CREATE TABLE IF NOT EXISTS KEYHOLDER(ID INTEGER PRIMARY KEY, KEY VARCHAR(255))")
        cursor.execute("INSERT OR IGNORE INTO KEYHOLDER(ID, KEY) VALUES(1, '')")
        cursor.execute("UPDATE KEYHOLDER SET KEY=? WHERE ID=1", (key,))
        conn.commit()
        print("New APIKEY: ", cursor.execute("SELECT KEY FROM KEYHOLDER WHERE ID=1").fetchone()[0])
        conn.close()



    def get_key(self):
        conn = sqlite3.connect('../sql.db')
        cursor = conn.cursor()
        row = cursor.execute("SELECT KEY FROM KEYHOLDER WHERE ID=1").fetchone()
        conn.close()
        return row

