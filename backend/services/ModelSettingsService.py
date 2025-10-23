from backend.repository.ModelSettingsRepository import ModelSettingsRepository


class ModelSettingsService:    
    
    def __init__(self):
        self.model_repository = ModelSettingsRepository()
    
    
    def change_settings(self,active_model,prompt_type,user_id):
        self.model_repository.change_settings_by_user_id(active_model,prompt_type,user_id)
        
        
    def delete_user(self,user_id):
        self.model_repository.delete_user_by_id(user_id)