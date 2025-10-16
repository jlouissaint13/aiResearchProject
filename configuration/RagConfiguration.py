
class RagConfiguration:

    EMBEDDING_MODEL = [

        "all-mpnet-base-v2",
        "intfloat/e5-large-v2"

                       ]

    LLM_MODEL = 'llama3.2'

    OLLAMA_API_URL = "http://localhost:11434"

    TOP_K = 7
    CHUNK_SIZE = 30
    CHUNK_OVERLAP = 1
    BATCH_SIZE = 64




