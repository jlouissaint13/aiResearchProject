from flask import Blueprint,jsonify,request
from backend.services.RegisterService import RegisterService
from backend.models.RegisterModel import RegisterModel
register_blueprint = Blueprint("user",__name__)




@register_blueprint.route("/register",methods=['POST'])
def register_request():
    registerService = RegisterService()
    data = request.get_json()

    role = data.get("role", "user")
    admin_key = data.get("admin_key", "")  # new field for admin signup key

    registerModel = RegisterModel(data.get("firstName"),data.get("email"),data.get("username"),data.get("password"))

    result = registerService.createAccount(registerModel, role, admin_key)

    if result == False:
        return jsonify({"error": "User already exists"}), 409
    elif result == "invalid_admin_key":
        return jsonify({"error": "Invalid admin key"}), 403
    elif result == "pending_approval":
        return jsonify({"message": "Admin account pending approval"}), 202
    
    

    return jsonify({"message": "Account created"}), 200
