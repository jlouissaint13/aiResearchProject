import google.generativeai as genai
import os
from dotenv import load_dotenv
from flask import Blueprint, jsonify

load_dotenv()

gemini_blueprint = Blueprint("gemini_controller", __name__)
GEMINI_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GEMINI_KEY)

@gemini_blueprint.route("/gemini-models", methods=["GET"])
def list_gemini_models():
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