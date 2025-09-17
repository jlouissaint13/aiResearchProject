import chromadb
import uuid
import pathlib



class ChromaManager:
    def __init__(self):
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.create_collection(name="my_collection")
        database_directory = pathlib.Path(__file__).parent / "chromadb"
        client = chromadb.PersistentClient(database_directory)

    def store(self,chunk_list):
        for i in range(len(chunk_list)):
            self.collection.add(
             ids=[str(chunk_list[i].id)],
             documents=[chunk_list[i].text],
             embeddings=[chunk_list[i].embedding]
            )
        print("data stored?" , self.collection.count())
    #clears db
    def reset(self):
        self.chroma_client.reset()
    #checks connection
    def heartbeat(self):
        self.chroma_client.heartbeat()

