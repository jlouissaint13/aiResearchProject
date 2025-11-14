
from backend.models.Chunk import Chunk
from sentence_transformers import SentenceTransformer
from configuration.RagConfiguration import RagConfiguration

class EmbedHandler:
    embed_model = SentenceTransformer('all-mpnet-base-v2')

    def embed(self, chunks):
        embedded = self.embed_model.encode(chunks,
                                            batch_size=RagConfiguration.BATCH_SIZE,
                                            convert_to_numpy=True,
                                            show_progress_bar=True,
                                            normalize_embeddings=True
                                           )
        return embedded.tolist()

    def get_chunk_embedding_list(self, chunks):
        embeddings = self.embed(chunks)

        chunk_embedding_list = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_embedding_list.append(Chunk(i, chunk, embedding))
        # temp id solution unique identifier and duplication prevention later
        return chunk_embedding_list
