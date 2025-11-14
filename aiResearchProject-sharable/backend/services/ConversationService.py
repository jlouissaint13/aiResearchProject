from backend.repository.ConversationRepository import ConversationRepository
from backend.repository.MessageRepository import MessageRepository
class ConversationService:
    def __init__(self):
        self.conversation_repository = ConversationRepository()
        self.message_repository = MessageRepository()
    def create_conversation(self,conversation_id,user_id,title):
        #after research this function is not needed because uuid probabilities
        #if self.conversation_repository.conversation_exists(conversation_id, user_id): 
           # return 'convo exists try again', 409
        
        
        self.conversation_repository.create_conversation(conversation_id,user_id,title)
        


    def get_conversations(self,user_id):
        return self.conversation_repository.select_conversations_by_id(user_id)
    
    
    def conversation_last_modified(self,conversation_id,user_id):
        self.conversation_repository.conversation_last_modified_at(conversation_id,user_id)
    
    
    def get_conversation_by_title(self,normalized_title,user_id):
        return self.conversation_repository.retrieve_converstion_id_by_title_and_user_id(normalized_title,user_id)
    
    def delete_conversation(self,conversation_id,user_id):
        self.conversation_repository.delete_conversation(conversation_id, user_id)
        self.message_repository.delete_all_messages_by_id(conversation_id,user_id)