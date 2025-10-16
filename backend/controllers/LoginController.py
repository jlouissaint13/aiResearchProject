from flask import Blueprint,request
from backend.models.LoginModel import LoginModel
from backend.services.LoginService import LoginService
login_blueprint = Blueprint("/login",__name__)
login_service = LoginService()


@login_blueprint.route("/auth",methods=['POST'])
def login_request():
    data = request.get_json()
    login_model = LoginModel(data.get("username"),data.get("password"))
    if login_service.auth(login_model.username,login_model.password) == 404:
        return "user does not exist",404
    elif login_service.auth(login_model.username,login_model.password) == 401:
        return "invalid password",401

    return "success",200
@login_blueprint.route("/user_logged",methods=['POST'])
def user_logged():
    data = request.get_json()
    username = data.get("username")
    user_id = login_service.get_user_id(username)
    name = login_service.get_first_name(username)
    response = {
        "userID" : user_id,
        "firstName" : name,
    }



    return  response, 200

