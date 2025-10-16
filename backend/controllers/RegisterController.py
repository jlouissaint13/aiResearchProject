from flask import Blueprint,jsonify,request
from backend.services.RegisterService import RegisterService
from backend.models.RegisterModel import RegisterModel
register_blueprint = Blueprint("user",__name__)




@register_blueprint.route("/register",methods=['POST'])
def register_request():
    registerService = RegisterService()
    data = request.get_json()

    registerModel = RegisterModel(data.get("firstName"),data.get("email"),data.get("username"),data.get("password"))
    if registerService.createAccount(registerModel) == False:
        return jsonify({"error": "User exists"}), 409

    return jsonify({"message": "Account created"}), 200
