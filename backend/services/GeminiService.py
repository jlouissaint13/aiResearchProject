import os

import google.generativeai as genai
import requests
from dotenv import load_dotenv, find_dotenv



class GeminiService:
    def __init__(self):
        pass


    def valid_key(self,key):
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"

        try:
            response = requests.get(url)
            response.raise_for_status()
            return True
        except Exception as e:
            return False


    def get_gemini_models(self):
        env_path = find_dotenv()
        load_dotenv(env_path, override=True)
        GEMINI_KEY = os.getenv("GOOGLE_API_KEY")
        #fix redundancy here later
        if not self.valid_key(GEMINI_KEY):
            return []
        all_models = []
        allowed_models = [
            "models/gemini-pro-latest",
            "models/gemini-flash-latest",
            "models/gemini-flash-lite-latest"
        ]
        #double call
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_KEY}"
        response = requests.get(url)
        response.raise_for_status()

        gemini_data = response.json()
        for model in gemini_data.get("models", []):
            if model.get("name") in allowed_models:
                all_models.append({
                    "id": model["name"],
                    "name": model.get("displayName", model["name"]),
                    "provider": "gemini"
                })

        return all_models

