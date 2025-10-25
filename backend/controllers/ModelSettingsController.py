import os

from dotenv import find_dotenv, load_dotenv
from flask import jsonify,request,Blueprint
from backend.services.OpenAIService import OpenAIService
from backend.services.GeminiService import GeminiService
from backend.services.ModelSettingsService import ModelSettingsService
model_settings_blueprint = Blueprint("model_settings",__name__)
model_settings_service = ModelSettingsService()


@model_settings_blueprint.route("/change_settings")
def change_settings():
    data = request.get_json()
    pass


@model_settings_blueprint.route('/retrieve_models',methods=['GET'])
def get_models():
    env_path = find_dotenv()
    load_dotenv(env_path, override=True)
    openai_service = OpenAIService()
    gemini_service = GeminiService()
    model_settings_service = ModelSettingsService()
    all_models = []

    openai_models = openai_service.get_openai_models()
    gemini_models = gemini_service.get_gemini_models()
    #add check later to see if this is actually installed
    llama_model = model_settings_service.llama_model()


    all_models = openai_models + gemini_models + llama_model


    return jsonify(all_models)
@model_settings_blueprint.route('/save_model_settings',methods=['POST'])
def save_model_settings():
    data = request.get_json()
    user_id = data.get("user_id")
    prompt_type = data.get("prompt_type")
    active_model = data.get("active_model")
    provider = data.get("provider")



    model_settings_service.change_settings(active_model,prompt_type,provider,user_id)
    return "success" , 200

    
@model_settings_blueprint.route('/retrieve_settings',methods=["POST"])
def get_model_settings():
    data = request.get_json()
    user_id = data.get("user_id")
    #I need to write code to make sure that the user still has access to their set model
    #an example would be their default model being an open ai one but then they remove access to that key
    response =  model_settings_service.retrieve_user_settings(user_id)

    return response, 200

