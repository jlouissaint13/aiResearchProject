import os

from dotenv import find_dotenv, load_dotenv
from flask import jsonify,request,Blueprint
from backend.services.OpenAIService import OpenAIService
from backend.services.GeminiService import GeminiService

model_settings_blueprint = Blueprint("model_settings",__name__)



@model_settings_blueprint.route("/change_settings")
def change_settings():
    data = request.get_json()
    pass


@model_settings_blueprint.route('/retrieve_models',methods=['GET'])
def get_models():
    env_path = find_dotenv()
    load_dotenv(env_path, override=True)
    OPENAI_KEY = os.getenv("OPENAI_API_KEY")
    GEMINI_KEY = os.getenv("GOOGLE_API_KEY")

    openai_service = OpenAIService()
    gemini_service = GeminiService()
    all_models = []

    openai_models = openai_service.get_openai_models()
    gemini_models = gemini_service.get_gemini_models()
    all_models = openai_models + gemini_models

    return jsonify(all_models)


    


