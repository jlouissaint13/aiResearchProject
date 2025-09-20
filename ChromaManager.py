import shutil

import chromadb
import pathlib



class ChromaManager:
    def __init__(self):

        database_directory = pathlib.Path(__file__).parent / "chromadb"
        self.client = chromadb.PersistentClient(path=database_directory)
        self.collection = self.client.get_or_create_collection(name="my_collection")

    def store(self,chunk_list):
        for i in range(len(chunk_list)):
            self.collection.add(
             ids=[str(chunk_list[i].id)],
             documents=[chunk_list[i].text],
             embeddings=[chunk_list[i].embedding]
            )
        print("data stored?" , self.collection.count())
    #deletes db
    def delete(self):
        database_directory = pathlib.Path(__file__).parent / "chromadb"
        if database_directory.exists():
            shutil.rmtree(database_directory)

    #checks connection
    def heartbeat(self):
        self.client.heartbeat()



    def check_db(self):
        results = self.collection.get(
            include=['documents','embeddings']
        )
        print(results['embeddings'])
        print(results['documents'])
