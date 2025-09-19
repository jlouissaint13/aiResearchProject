
from ChromaManager import ChromaManager
from Chunk import Chunk
from sentence_transformers import SentenceTransformer
from RagConfiguration import RagConfiguration

class EmbedHandler:
    embed_model = SentenceTransformer(RagConfiguration.EMBEDDING_MODEL[0])

    def embed(self, chunks):
        embedded = self.embed_model.encode(chunks)
        return embedded.tolist()

    def store_chunks_and_embeddings(self, chunks):
        chroma_manager = ChromaManager()
        embeddings = self.embed(chunks)

        chunk_list = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_list.append(Chunk(i, chunk, embedding))
        # temp id solution unique identifier and duplication prevention later
        chroma_manager.store(chunk_list)
