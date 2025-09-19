from langchain.text_splitter import MarkdownTextSplitter
import nltk

from nltk.tokenize import sent_tokenize

from RagConfiguration import RagConfiguration

class ChunkSplitter:

    def __init__(self):
        self.markdown_splitter = MarkdownTextSplitter()
        self.chunk_size = RagConfiguration.CHUNK_SIZE
        self.chunk_overlap = RagConfiguration.CHUNK_OVERLAP
        self.nltk_check()

    def nltk_check(self):
        # keep console clean annoying download message on init everytime other wise
        for i in ['punkt', 'punkt_tab']:
            try:
                nltk.data.find(f'tokenizers/{i}')
            except Exception:
                nltk.download(i)

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


