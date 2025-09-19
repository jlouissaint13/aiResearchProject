from langchain.chains.llm import LLMChain
from sympy.polys.polyconfig import query

from ChromaManager import ChromaManager
from sentence_transformers import SentenceTransformer
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

from RagConfiguration import RagConfiguration

class Rag:
    embed = SentenceTransformer(RagConfiguration.EMBEDDING_MODEL[0])
    def __init__(self,chroma_manager,query):
        self.chroma = chroma_manager

        self.query = query


    def get_query(self):
        return self.query

    def fetch_query_response(self):
        query_embedding = self.embed.encode([self.query]).tolist()[0]

        results = self.chroma.collection.query(
            query_embeddings=[query_embedding],
            n_results = RagConfiguration.TOP_K
        )

        retrieved_docs = results["documents"][0]
        return retrieved_docs

    def context_Text(self):
        context_text = "\n\n".join(self.fetch_query_response())

        return context_text

    def model_run(self):
        model = OllamaLLM(model=RagConfiguration.LLM_MODEL)

        chain = Rag.prompt_builder() | model

        return chain.invoke({
            "query": self.query,
            "context": self.context_Text()
        })

    @staticmethod
    def prompt_builder():
        template = """You are a research assistant.
        Here is some relevant info: {context}
        Here is the question: {query}"""
        return ChatPromptTemplate.from_template(template)
