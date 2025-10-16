
from flask import Blueprint, request, jsonify

from backend.services.MessageService import MessageService
from backend.services.RagService import RagService
from backend.services.ConversationService import ConversationService
from backend.services.ConversationService import ConversationService
message_service = MessageService()
rag_service = RagService()
conversation_service = ConversationService()

message_blueprint = Blueprint('message',__name__)






@message_blueprint.route("/get_messages_by_conversation_id",methods=['POST'])
def get_all_messages():
    data = request.get_json()
    user_id = data.get("user_id")
    conversation_id = data.get("conversation_id")
    messages = message_service.get_messages_by_id(user_id,conversation_id)
    return jsonify(messages),200

@message_blueprint.route("/send_message",methods=['POST'])
def message_handler():
    
    data = request.get_json()
    user_content = data.get('content')
    role = data.get('sender')
    message_id = data.get("message_id")
    user_id = data.get("user_id")
    conversation_id = data.get("conversation_id")
    
    message_service.send_user_message(message_id, conversation_id, user_content, role, user_id)
    
    
                                          
    response = message_service.send_model_message(conversation_id,user_id,user_content)

    conversation_service.conversation_last_modified(conversation_id,user_id)


    return response
    
