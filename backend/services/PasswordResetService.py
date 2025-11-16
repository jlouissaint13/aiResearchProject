
import sqlite3
from backend.services.RegisterService import RegisterService

from datetime import datetime, timedelta


class PasswordResetService:
        def __init__(self):
            from backend.repository.PasswordResetRepository import UserRepository
            self.user_repo = UserRepository()
            self.conn = sqlite3.connect("sql.db", check_same_thread=False)
            self.cursor = self.conn.cursor()
            self.register_service = RegisterService()

        def verify_code(self, email, code):
            record = self.user_repo.get_reset_code(email)
            if not record:
                return False

            saved_code, expires_at = record
            if saved_code != code:
                return False

            if datetime.utcnow() > datetime.fromisoformat(expires_at):
                return False

            return True

        def update_password(self, email, new_password):
            new_password = self.register_service.password_hash(new_password)


            return self.user_repo.update_password_by_email(email,new_password)





