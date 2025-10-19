
import pymupdf4llm
from backend.services.ChunkSplitter import ChunkSplitter
from backend.services.EmbedHandler import EmbedHandler
from backend.services.MetadataService import MetadataServie
from backend.repository.ChromaRepository import ChromaRepository
class ChromaService:
    def __init__(self):
        self.chunkSplitter = ChunkSplitter()
        self.embedHandler = EmbedHandler()
        self.chroma_repository = ChromaRepository()

    def store_pdf_chroma(self, pdf_path, file_name):
        metadata = MetadataServie(pdf_path)
        metadata_info = metadata.get_metadata_dto(file_name)
        
        if self.hash_exists(metadata_info):
            print("hash exists but may be on different account so we won't return 409 anymore")
            return None 
        
        
        md_text = pymupdf4llm.to_markdown(pdf_path)
        print("pdf converted text")
        chunks = self.chunkSplitter.semantic_split(md_text)

        chunk_embed_list = self.embedHandler.get_chunk_embedding_list(chunks)

        return self.chroma_repository.store(chunk_embed_list, metadata_info)



    def hash_exists(self, metadata):
        return self.chroma_repository.hash_exists(metadata.hash_value)
    
    
    
    def delete_from_chroma_db(self,hash_value):
        self.chroma_repository.delete_chunks_by_hash_value(hash_value)