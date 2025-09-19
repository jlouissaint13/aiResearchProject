#not being used right now but later I want the user to be able to choose by speed or accuracy which is why I have a list of models instead of just one
from langchain.prompts import ChatPromptTemplate
class RagConfiguration:

    EMBEDDING_MODEL = [

        "all-mpnet-base-v2",
        "intfloat/e5-large-v2"

                       ]

    LLM_MODEL = 'llama3.2'

    OLLAMA_API_URL = "http://localhost:11434"

    TOP_K = 2
    CHUNK_SIZE = 10
    CHUNK_OVERLAP = 1

