import shutil

import chromadb
import pathlib



class ChromaRepository:
    def __init__(self):

        database_directory = pathlib.Path(__file__).parent / "chromadb"
        self.client = chromadb.PersistentClient(path=database_directory)
        self.collection = self.client.get_or_create_collection(name="my_collection")

    def store(self, chunk_embed_list, metadata_dto):
        

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
                "file_name": metadata_dto.file_name,
                
                
                
        })

    # Batch insert
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
    )

        print("data stored?", self.collection.count())
        return 200

    
    def delete_chunks_by_hash_value(self,hash_value,ref_count):
        
        if ref_count == 0:
            print(ref_count,"deleted")
            self.collection.delete(where={"hash_value": hash_value})
            return 
        print("not removing pdf user still referencing it",ref_count)


    def query_results_logged_in(self,query_embedding,top_k,user_pdfs):
        return self.collection.query(
           query_embeddings=[query_embedding],
           n_results = top_k,
            where={
                "hash_value": {
                    "$in": user_pdfs
                }
            }
       )
    
    
    def query_results_guest(self,query_embedding,top_k=5):
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results = top_k,
        )
    
    

    def hash_exists(self, hash_value):
        result = self.collection.get(where={"hash_value": hash_value})
        if not result["ids"]:
            return False

        return True



        #deletes db
    def delete_database(self):
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
