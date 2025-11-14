from backend.repository.LoginRepository import LoginRepository

class LoginService:
    def __init__(self):
        self.login_repository = LoginRepository()

    def auth(self,username,password):
        if not self.login_repository.user_exists(username):
            return 404

        if not self.login_repository.check_pw(username, password):
            return 401
        
        if not self.login_repository.is_active(username):
           return 403
       


        return 200


    def get_user_id(self,username):
        return self.login_repository.get_id(username)



    def get_first_name(self,username):
       return self.login_repository.get_name(username)




