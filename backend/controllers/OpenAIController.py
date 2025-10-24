from flask import Blueprint, jsonify
from openai import OpenAI
import os
from dotenv import load_dotenv, find_dotenv


openai_blueprint = Blueprint("openai_controller", __name__)


@openai_blueprint.route("/models", methods=["GET"])
def list_openai_models():


    env_path = find_dotenv()
    load_dotenv(env_path, override=True)
    OPENAI_KEY = os.getenv("OPENAI_API_KEY")


    if OPENAI_KEY is None:
        return "chatgpt not active" , 404

    client = OpenAI(api_key=OPENAI_KEY)


    try:
        response = client.models.list()
        allowed_models = {
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
        chat_model_ids = [model.id for model in response.data if model.id in allowed_models]
        return jsonify(chat_model_ids)
    except Exception as e:
        return jsonify({"error": str(e)}), 500