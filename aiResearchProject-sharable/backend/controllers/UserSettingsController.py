
from flask import Blueprint,jsonify, request

from backend.models.UserSettingsModel import UserSettingsModel
from backend.services.UserSettingsService import UserSettingsService

user_settings_blueprint = Blueprint("user_settings",__name__)
user_settings_service = UserSettingsService()

@user_settings_blueprint.route("/update_user_info",methods=['PATCH'])
def update_user_info():
    data = request.get_json()
    user_id = data.get("user_id")
    email = data.get("email")
    username = data.get("username")
    password = data.get("password")
    user_settings_dto = UserSettingsModel(user_id,email,username,password)
    return "success", user_settings_service.update_user_info(user_settings_dto)

    
    
@user_settings_blueprint.route("/retrieve_user_info",methods=['POST']) 
def get_user_info():
    data = request.get_json()
    user_id = data.get("user_id")
    response = user_settings_service.get_user_info(user_id)
    return jsonify(response),200


@user_settings_blueprint.route("/delete/user_account",methods=['DELETE'])
def delete_user():
    data = request.get_json()
    user_id = data.get('user_id')
    user_settings_service.delete_user_information(user_id)
    return "success", 200
