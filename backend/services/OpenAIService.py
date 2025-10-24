import os

from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

class OpenAIService:
    def __init__(self):
        pass


    def valid_key(self,key):

        try:
            client = OpenAI(api_key=key)

            client.models.list()
            return True
        except Exception as e:
            return False


    def get_openai_models(self):
        env_path = find_dotenv()
        load_dotenv(env_path, override=True)
        OPENAI_KEY = os.getenv("OPENAI_API_KEY")
        all_models = []
        if not self.valid_key(OPENAI_KEY):
            return
        allowed_models_gpt = {
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
            "gpt-4",
            "gpt-4-32k",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-chat",
        }
        client = OpenAI(api_key=OPENAI_KEY)
        for model in client.models.list():
            if model.id in allowed_models_gpt:
                all_models.append({
                    "id": model.id,
                    "name": model.id,
                    "provider": "openai"
                })
        return all_models

