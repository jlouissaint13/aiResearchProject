from langchain.text_splitter import MarkdownTextSplitter
import nltk
from langchain_text_splitters import RecursiveCharacterTextSplitter
from onnxruntime.tools.offline_tuning import embed
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
from ChromaManager import ChromaManager
from Chunk import Chunk


class ChunkSplitter:
    def __init__(self):
        self.markdown_splitter = MarkdownTextSplitter()

       # self.semantic_splitter = RecursiveCharacterTextSplitter(
        #    chunk_size=1000,
         #   chunk_overlap=200
        #)
        self.chunk_size = 10
        self.chunk_overlap = 1
        nltk.download('punkt')
        nltk.download("punkt_tab")
        self.model = SentenceTransformer('all-mpnet-base-v2')

    def semantic_split(self,pdf):
        markdown_pdf = self.markdown_splitter.split_text(pdf)
        text = " ".join(markdown_pdf).lower()
        sentences = sent_tokenize(text)
        chunks = []
        i = 0
        while i< len(sentences):
            chunk = sentences[i:i + self.chunk_size]
            chunks.append(" ".join(chunk))
            i += self.chunk_size - self.chunk_overlap

        return chunks

    def embed(self,chunks):
        embedded = self.model.encode(chunks)
        return embedded.tolist()

    def enumurate_chunks(self,pdf):
        chromamanager = ChromaManager()
        chunks = self.semantic_split(pdf)
        embeddings = self.embed(chunks)
        #combination = [
           # {"id": i, "text": chunk, "embedding": embedding}
          #  for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        #]
        chunk_list = []
        for i, (chunk,embedding) in enumerate(zip(chunks,embeddings)):
            chunk_list.append(Chunk(i,chunk,embedding))
           #temp id solution unique identifier and duplication prevention later
        chromamanager.store(chunk_list)

