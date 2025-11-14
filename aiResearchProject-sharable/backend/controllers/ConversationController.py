from flask import Blueprint,request

from backend.services.ConversationService import ConversationService

conversation_service = ConversationService()
conversation_blueprint = Blueprint('conversation',__name__)

@conversation_blueprint.route('/receive',methods=['POST'])
def create_new_conversation():
    data = request.get_json()
    conversation_id = data.get("conversation_id")
    user_id = data.get("user_id")
    title = data.get("title")
      
      
    if user_id is not None: 
      conversation_service.create_conversation(conversation_id,user_id,title)
      return "success" , 200
        

    return "success", 200

@conversation_blueprint.route("/get_conversations_by_id",methods=['POST'])
def get_conversation_by_id():
    data = request.get_json()
    return conversation_service.get_conversations(data.get("user_id")),200




@conversation_blueprint.route("/delete_conversation",methods=['DELETE'])
def delete_conversation_by_id():
    data = request.get_json()
    conversation_id = data.get("conversation_id")
    user_id = data.get("user_id")
    conversation_service.delete_conversation(conversation_id,user_id)
    
    return "success" , 200


