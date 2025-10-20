
import pymupdf4llm
from backend.services.ChunkSplitter import ChunkSplitter
from backend.services.EmbedHandler import EmbedHandler
from backend.services.MetadataService import MetadataServie
from backend.repository.ChromaRepository import ChromaRepository
from backend.services.PDFManagerService import PDFManagerService
class ChromaService:
    def __init__(self):
        self.chunkSplitter = ChunkSplitter()
        self.embedHandler = EmbedHandler()
        self.chroma_repository = ChromaRepository()
        self.pdfManager = PDFManagerService()
    def store_pdf_chroma(self, pdf_path, file_name,user_id):
        metadata = MetadataServie(pdf_path)
        metadata_info = metadata.get_metadata_dto(file_name)
        
        
        md_text = pymupdf4llm.to_markdown(pdf_path)
        print("pdf converted text")
        chunks = self.chunkSplitter.semantic_split(md_text)

        chunk_embed_list = self.embedHandler.get_chunk_embedding_list(chunks)

        return self.chroma_repository.store(chunk_embed_list, metadata_info)



    def hash_exists(self, metadata):
        return self.chroma_repository.hash_exists(metadata.hash_value)
    
    
    
    def delete_from_chroma_db(self,hash_value):
        ref_count = self.pdfManager.get_ref_count(hash_value)
        
        self.chroma_repository.delete_chunks_by_hash_value(hash_value,ref_count)
        
        
    