
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

    def store_file_chroma(self,pdf_path):
        metadata = MetadataServie(pdf_path)
        metadata_info = metadata.get_metadata_dto()

        md_text = pymupdf4llm.to_markdown(pdf_path)
        print("pdf converted text")
        chunks = self.chunkSplitter.semantic_split(md_text)

        chunk_embed_list = self.embedHandler.get_chunk_embedding_list(chunks)

        self.chroma_repository.store(chunk_embed_list, metadata_info)




