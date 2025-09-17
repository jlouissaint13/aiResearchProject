from langchain.text_splitter import MarkdownTextSplitter
import nltk
from langchain_text_splitters import RecursiveCharacterTextSplitter
from onnxruntime.tools.offline_tuning import embed
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
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
        self.model = SentenceTransformer('all-mpnet-base-v2')

    def semantic_split(self,pdf):
        markdown_pdf = self.markdown_splitter.split_text(pdf)
        sentences = sent_tokenize(markdown_pdf)
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
        chunks = self.semantic_split(pdf)
        embedded = self.embed(chunks)
