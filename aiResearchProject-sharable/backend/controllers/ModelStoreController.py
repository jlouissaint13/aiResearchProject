from flask import jsonify,request,Blueprint

from backend.services.ModelStoreService import ModelStoreService

store_model_blueprint = Blueprint("store_model",__name__)


#WE WILL NOT STORE ACTUAL KEY IN DB
@store_model_blueprint.route('/store',methods=['POST'])
def store():
    data = request.get_json()
    provider = data.get('provider')
    key = data.get('key')
    key_name = data.get('key_name')



    model_store_services = ModelStoreService()
    try:
     model_store_services.validate_key(provider,key)
    except Exception as e:
        return "invalid key",400

    model_store_services.store_info_env(provider,key)
    model_store_services.store_info_database(provider,key_name)

    return "success" , 200