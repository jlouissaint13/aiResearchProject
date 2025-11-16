
from backend.repository.ModelSettingsRepository import ModelSettingsRepository
from backend.services.PDFManagerService import PDFManagerService


class ModelSettingsService:    
    
    def __init__(self):
        self.model_repository = ModelSettingsRepository()
        self.pdf_manager_service = PDFManagerService()
    def change_settings(self,active_model,prompt_type,provider,user_id):
        self.model_repository.change_settings_by_user_id(active_model,prompt_type,provider,user_id)
        
        
    def delete_user(self,user_id):
        self.model_repository.delete_settings_by_id(user_id)


    def retrieve_user_settings(self,user_id):


       tupleReturned =  self.model_repository.retrieve_user_settings_by_user_id(user_id)
       if tupleReturned is None:
           return None

       model_settings = { "activeModel" : tupleReturned[0],
                   "promptType" : tupleReturned[1],
                   "provider" : tupleReturned[2]
       }
       return model_settings



    def llama_model(self):
        list = [{
            "id": "llama3.2",
            "name": "Llama 3.2",
            "provider": "meta"
        }]

        return list


    def data_visualization_allowed(self,user_id):


        model_settings = self.retrieve_user_settings(user_id)
        provider = model_settings['provider']
        total_documents = len(self.pdf_manager_service.get_searchable_documents(user_id))
        if total_documents == 0 or provider != 'openai':
            return False

        return True

