from flask import jsonify,request,Blueprint


model_settings_blueprint = Blueprint("model_settings",__name__)



@model_settings_blueprint.route("/change_settings")
def change_settings():
    data = request.get_json()
    pass




