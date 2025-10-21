from backend.repository.UserSettingsRepository import UserSettingsRepository
from backend.models.UserSettingsModel import UserSettingsModel
class UserSettingsService:
    def __init__(self):
        self.user_settings_repository = UserSettingsRepository()
        
    def update_user_info(self,user_settings_dto:UserSettingsModel):
        user_id = user_settings_dto.user_id
        email = user_settings_dto.email
        username = user_settings_dto.username
        password = user_settings_dto.password

        if self.user_exists(user_id,email,username):
            return 409
        
        
        if len(password) != 0:
            self.user_settings_repository.update_user_info_by_user_id_pw(user_id, email, username, password)
            return 200
        self.user_settings_repository.update_user_info_by_user_id_no_pw(user_id,email,username)
        return 200
    
    
    def user_exists(self,user_id,email,username):
        return self.user_settings_repository.user_exists(user_id,email,username)
    
    
    def get_user_info(self,user_id):
       return dict(self.user_settings_repository.get_user_info_for_settings_page_by_user_id(user_id))
    
    def delete_user_information(self,user_id):
        self.user_settings_repository.delete_user_data_by_user_id(user_id)