import bcrypt

from backend.models.RegisterModel import RegisterModel
from backend.repository.RegisterRepository import RegisterRepository
import sqlite3

class RegisterService:
    def __init__(self):
        self.registerRepository = RegisterRepository()
        self.DB_PATH = "sql.db"

    def createAccount(self, register_model:RegisterModel, role: str = "user", admin_key: str = ""):
        firstName = register_model.first_name
        email = register_model.email
        username = register_model.username

        if self.user_exists(email,username):
            return False

        password = self.password_hash(register_model.password)

        # If registering as admin
        if role.lower() == "admin":
            if admin_key.lower() != "athena":
                # Invalid admin key — registration fails
                return "invalid_admin_key"
            
          # Valid key — store in PENDING_ADMINS table for approval
            with sqlite3.connect(self.DB_PATH) as conn:
                conn.execute("""
                        INSERT INTO PENDING_ADMINS (username, email, first_name, password)
                        VALUES (?, ?, ?, ?)
                     """, (username, email, firstName, password))
                conn.commit()
            return "pending_approval"

        self.registerRepository.create_account(firstName, email, username, password, role)
        return True




    def password_hash(self,password):
        password_bytes = password.encode("utf-8")

        salt = bcrypt.gensalt()

        pw_hash = bcrypt.hashpw(password_bytes,salt)
        hashed_string = pw_hash.decode("utf-8")

        return hashed_string
    
    
    def user_exists(self,email,username):
        return self.registerRepository.user_exists(email,username)