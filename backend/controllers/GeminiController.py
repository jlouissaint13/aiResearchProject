import google.generativeai as genai
import os
from dotenv import load_dotenv, find_dotenv
from flask import Blueprint, jsonify


gemini_blueprint = Blueprint("gemini_controller", __name__)



@gemini_blueprint.route("/gemini-models", methods=["GET"])
def list_gemini_models():

    env_path = find_dotenv()
    load_dotenv(env_path, override=True)
    GEMINI_KEY = os.getenv("GOOGLE_API_KEY")
    if GEMINI_KEY is None:
        return "gemini not active" , 404
    genai.configure(api_key=GEMINI_KEY)
    
    try:
        all_models = genai.list_models()
        main_models = [
            "models/gemini-pro-latest",
            "models/gemini-flash-latest",
            "models/gemini-flash-lite-latest"
        ]
        chat_model_names = [model.name for model in all_models if model.name in main_models]
        return jsonify(chat_model_names)
    except Exception as e:
        return jsonify({"error": str(e)}), 500