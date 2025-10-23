
from sentence_transformers import SentenceTransformer
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from backend.repository.ChromaRepository import ChromaRepository
from configuration.RagConfiguration import RagConfiguration
from backend.services.PDFManagerService import PDFManagerService
chroma = ChromaRepository()

class RagService:
    embed = SentenceTransformer(RagConfiguration.EMBEDDING_MODEL[0])
    def __init__(self):
       self.chroma = chroma
       self.pdfManagerService = PDFManagerService()

    def get_query(self):
        return self.query

    def fetch_query_response(self):
        query_embedding = self.embed.encode([self.query]).tolist()[0]
        
        user_accessible_pdfs = self.pdfManagerService.get_searchable_documents(self.user_id)
        
        #if the list is empty that means the user hasn't entered any pdfs lets skip the chromadb check
        if len(user_accessible_pdfs) == 0 and self.user_id is not None:
            return ""
        
        if self.user_id is not None:
            results = self.chroma.query_results_logged_in(query_embedding,RagConfiguration.TOP_K,user_accessible_pdfs)
        else:
            results = self.chroma.query_results_guest(query_embedding)
        
        retrieved_docs = results["documents"][0]
        return retrieved_docs

    def context_Text(self):
        
        context_text = "\n".join(self.fetch_query_response())
        return context_text

    def model_run(self):
        model = OllamaLLM(model=RagConfiguration.LLM_MODEL)

        chain = RagService.prompt_builder() | model
        
        if self.context_Text() == "":
            return "Please insert a PDF I don't have any data to reference."
        
        
        
        return chain.invoke({
            "query": self.query,
            "context": self.context_Text()
        })
    
    def response(self,text,user_id):
        self.query = text
        self.user_id = user_id
        answer = self.model_run()
        print(answer)
        return answer

    @staticmethod
    def prompt_builder():
        template = """
You are a **Creative Analyst**.

Draw insights, connections, or analogies from the CONTEXT to provide a thoughtful, imaginative, and engaging response.
Stay grounded in the text, but feel free to interpret or synthesize ideas meaningfully.
If you speculate, label it as (speculative).

CONTEXT:
{context}

QUESTION:
{query}

RESPONSE:
"""
        return ChatPromptTemplate.from_template(template)





