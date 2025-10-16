import shutil

import chromadb
import pathlib



class ChromaRepository:
    def __init__(self):

        database_directory = pathlib.Path(__file__).parent / "chromadb"
        self.client = chromadb.PersistentClient(path=database_directory)
        self.collection = self.client.get_or_create_collection(name="my_collection")

    def store(self, chunk_embed_list, metadata_dto):
        if self.hash_exists(metadata_dto.hash_value):
            print("PDF is already stored!")
            return

        ids = []
        documents = []
        embeddings = []
        metadatas = []

        for i in range(len(chunk_embed_list)):
            chunk = chunk_embed_list[i]
            ids.append(f"{metadata_dto.hash_value}_{i}")
            documents.append(chunk.text)
            embeddings.append(chunk.embedding)
            metadatas.append({
                "title": metadata_dto.title,
                "author": metadata_dto.author,
                "created_at": metadata_dto.created_at,
                "modified_at": metadata_dto.modified_at,
                "hash_value": metadata_dto.hash_value,
                "file_name": metadata_dto.file_name
        })

    # Batch insert
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
    )

        print("data stored?", self.collection.count())





    def hash_exists(self,hash_value):
        result = self.collection.get(where={"hash_value": hash_value})
        if len(result["ids"]) == 0:
            return False
        return True



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
