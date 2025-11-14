
from sentence_transformers import SentenceTransformer
from langchain_ollama.llms import OllamaLLM
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
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

        match self.provider:
            case 'openai':
                model = ChatOpenAI(model=self.active_model)
                print("openai used")

            case 'gemini':
                model = ChatGoogleGenerativeAI(model=self.active_model)
                print("gemini used")

            case 'meta':
                model = OllamaLLM(model=self.active_model)
                print("llama used")


        chain = RagService.prompt_builder(self.prompt_type) | model
        
        if self.context_Text() == "":
            return "Please insert a PDF I don't have any data to reference."
        
        
        
        return chain.invoke({
            "query": self.query,
            "context": self.context_Text()
        })
    
    def response(self,text,active_model,prompt_type,provider,user_id):
        self.query = text
        self.user_id = user_id
        self.active_model = active_model
        self.prompt_type = prompt_type
        self.provider = provider

        self.data_visualization_mode()

        answer = self.model_run()
        print(answer)
        return answer



    def data_visualization_mode(self):
        if self.prompt_type == "data-visualization":
            print("data visualization mode")
            self.query += """Return JSON in this format:
        {{
            "chart_type": "bar" | "line" | "scatter",
            "data": [
                {{"column_1_name": value, "column_2_name": value, ...}},
                {{"column_1_name": value, "column_2_name": value, ...}}
            ]
        }}
        
        For example, for a simple bar chart of two patients' VCN, return:
        {{"chart_type": "bar", "data": [{"patient": "Patient 1", "vcn": 1.5}, {"patient": "Patient 2", "vcn": 2.1}]}}
        """


    @staticmethod
    def prompt_builder(prompt_type):
        template = get_research_prompt(prompt_type)

        return ChatPromptTemplate.from_template(template)







def get_research_prompt(mode):

    if mode == "deep-research":
        return """
You are a **Research Analyst**.

Your job is to produce a thorough, structured, and deeply reasoned response using only the information provided in the CONTEXT.
Use clear logic, reference multiple parts of the text, and explain relationships or causes where relevant.

If data is missing, acknowledge it — do not invent details.

CONTEXT:
{context}

QUESTION:
{query}

RESPONSE:
"""

    elif mode == "short-and-sweet":
        return """
You are a **Concise Research Assistant**.

Answer the QUESTION briefly and clearly using only the CONTEXT.
Focus on accuracy and brevity. Avoid speculation, repetition, or filler phrases.

CONTEXT:
{context}

QUESTION:
{query}

RESPONSE:
"""

    elif mode == "creative":
        return """
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
    elif mode == "data-visualization":
        return """
    You are a **JSON Extraction Analyst**.
    Your response **MUST** be a single, valid JSON object and nothing else.
    Do NOT write any explanations or conversational text.

    Analyze the CONTEXT to answer the QUESTION.
    
    The QUESTION itself contains the **full instructions** and the
    **required JSON format**. Follow the instructions in the
    QUESTION field exactly.

    CONTEXT:
    {context}

    QUESTION:
    {query}

    JSON_RESPONSE:
    """
#if blank just return deep
    else:
       return """
You are a **Research Analyst**.

Your job is to produce a thorough, structured, and deeply reasoned response using only the information provided in the CONTEXT.
Use clear logic, reference multiple parts of the text, and explain relationships or causes where relevant.

If data is missing, acknowledge it — do not invent details.

CONTEXT:
{context}

QUESTION:
{query}

RESPONSE:
"""


