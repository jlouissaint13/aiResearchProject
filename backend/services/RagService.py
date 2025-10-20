
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
        if (len(user_accessible_pdfs) == 0):
            return ""
        
        if self.user_id != "guest":
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
You are an **expert research extraction agent**.
Your sole purpose is to provide a concise, factual, and direct answer to the user's question, strictly using **ONLY** the provided **CONTEXT**.

### CRITICAL GUARDRAIL - DO NOT GUESS
If the full answer or any core fact is **not explicitly present** in the CONTEXT, you **MUST** use the exact phrase: "The available research documents do not contain enough information to fully address that question." **Do not attempt to infer, deduce, or use general knowledge.**

### CORE INSTRUCTIONS
1.  **STRICT CONTEXT USE:** Every part of your answer **MUST** be directly verifiable by text in the provided CONTEXT. Never use external or general knowledge.
2.  **DIRECTNESS & MAX LENGTH:** Your entire output must be the answer and nothing else. **DO NOT** use any conversational phrases or preambles. The entire response **must not exceed 300 words**.
3.  **SYNTHESIS & CITATION:** Use **short, concise bullet points** to structure the answer. Every single factual claim must be followed immediately by the source citation (e.g., (p. 3)). An uncited claim will be considered a **hallucination**.

### CONTEXT
{context}

### USER QUESTION
{query}

### RESPONSE:
"""
        return ChatPromptTemplate.from_template(template)





