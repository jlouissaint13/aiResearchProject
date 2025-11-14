from langchain_core.messages import BaseMessage

from backend.services.RagService import RagService
from backend.repository.MessageRepository import MessageRepository
from backend.services.ModelSettingsService import ModelSettingsService
import uuid
class MessageService:
    def __init__(self):
        self.message_repository = MessageRepository()
        self.rag_service = RagService()
        self.model_settings = ModelSettingsService()
    def send_user_message(self, message_id, conversation_id, content, role, user_id,logged_in):
       
       
       if logged_in == 'true':
          self.message_repository.insert_message_by_id(message_id, conversation_id, content, role, user_id)
        
        
        
       
    
    def get_messages_by_id(self,user_id,conversation_id):
       return self.message_repository.retrieve_all_messages_by_id(user_id,conversation_id)
       
       
    def send_model_message(self, conversation_id,user_id,user_content,logged_in):
        model_message_id = str(uuid.uuid4())
        role = 'model'


        if logged_in == 'true':
            current_user_settings = self.model_settings.retrieve_user_settings(user_id)
            active_model = current_user_settings.get("activeModel")
            prompt_type = current_user_settings.get("prompt_type")
            provider = current_user_settings.get("provider")
        else:
            active_model = "llama3.2"
            prompt_type = "deep-research"
            provider = "meta"



        
        model_response = self.rag_service.response(user_content,active_model,prompt_type,provider,user_id)

        #So the local model returns the content as text
        #The external apis don't do this so we need to check the object type before we store it in sql
        #sql cannot handle the complex type so we need to convert if not text
        if isinstance(model_response, BaseMessage):
            model_response = model_response.content

        if logged_in == 'true':
            self.message_repository.insert_message_by_id(model_message_id,conversation_id,model_response,role,user_id)
            
        
        
        response = {
            "content" : model_response,
            "conversationID" : conversation_id,
            "role" : "model",
            "user_id": user_id,
            "message_id" : model_message_id
        }
        
        return response
        