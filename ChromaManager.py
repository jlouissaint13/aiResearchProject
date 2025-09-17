import chromadb

class ChromaManager:
    def __init__(self):

        chroma_client = chromadb.Client()
        collection = chroma_client.create_collection(name="my_collection")
        collection.add()
    def store(self,):