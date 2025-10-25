import os
from dotenv import set_key, load_dotenv, find_dotenv
from backend.repository.StoreModelRepository import StoreModelRepository
from backend.services.OpenAIService import OpenAIService
from backend.services.GeminiService import GeminiService
class ModelStoreService:
    def __init__(self):
        pass

    def store_info_database(self,provider,key_name):
        store_repo = StoreModelRepository()
        store_repo.store_model(provider,key_name)


    def store_info_env(self,provider,api_key):

        ENV_PATH = find_dotenv()

        load_dotenv(ENV_PATH)

        if provider == "openai":
            env_target = "OPENAI_API_KEY"
        elif provider == "gemini":
            env_target = "GOOGLE_API_KEY"

        set_key(ENV_PATH, env_target, api_key)

        os.environ[env_target] = api_key



    def validate_key(self,provider,key):
        gemini_service = GeminiService()
        openai_service = OpenAIService()

        if provider == 'openai':
            status = openai_service.valid_key(key)
        elif provider == "gemini":
            status = gemini_service.valid_key(key)


        if status == False:
            raise Exception("Invalid Key")

        return
