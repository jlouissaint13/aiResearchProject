import bcrypt

from backend.models.RegisterModel import RegisterModel
from backend.repository.RegisterRepository import RegisterRepository

class RegisterService:
    def __init__(self):
        self.registerRepository = RegisterRepository()

    def createAccount(self, register_model:RegisterModel):
        firstName = register_model.first_name
        email = register_model.email
        username = register_model.username

        if self.registerRepository.user_exists(register_model.email, register_model.username):
            return False

        password = self.password_hash(register_model.password)
        self.registerRepository.create_account(firstName, email, username, password)
        return True




    def password_hash(self,password):
        password_bytes = password.encode("utf-8")

        salt = bcrypt.gensalt()

        pw_hash = bcrypt.hashpw(password_bytes,salt)
        hashed_string = pw_hash.decode("utf-8")

        return hashed_string