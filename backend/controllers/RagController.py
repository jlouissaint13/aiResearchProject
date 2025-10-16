
from flask import Blueprint,jsonify,request
from backend.services.RagService import RagService

rag_blueprint = Blueprint('rag',__name__)

rag_service = RagService()


@rag_blueprint.route('/receive',methods=['POST'])
def receive_query():
    data = request.get_json()
    response = {
        "text" : rag_service.response(data.get('text')).strip(),
        "sender" : "model",
        "id" : "1",
    }

    return response, 200




