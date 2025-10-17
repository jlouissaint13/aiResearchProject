from backend.services.RagService import RagService 
from backend.repository.MessageRepository import MessageRepository
from backend.services.ConversationService import ConversationService
import uuid
class MessageService:
    def __init__(self):
        self.message_repository = MessageRepository()
        self.rag_service = RagService()
        
    def send_user_message(self, message_id, conversation_id, content, role, user_id,logged_in):
        if logged_in == 'true':
            print("saving messages user")
            self.message_repository.insert_message_by_id(message_id, conversation_id, content, role, user_id)
        
        
        
       
    
    def get_messages_by_id(self,user_id,conversation_id):
       return self.message_repository.retrieve_all_messages_by_id(user_id,conversation_id)
       
       
    def send_model_message(self, conversation_id,user_id,user_content,logged_in):
        model_message_id = str(uuid.uuid4())
        
        
        role = 'model'
        content = self.rag_service.response(user_content)
        if logged_in == 'true':
            print("saving model messages user")
            self.message_repository.insert_message_by_id(model_message_id,conversation_id,content,role,user_id) 
        
        
        
        response = {
            "content" : content,
            "conversationID" : conversation_id,
            "role" : "model",
            "user_id": user_id,
            "message_id" : model_message_id
        }
        
        return response
        