import random

from backend.services.PasswordResetService import PasswordResetService
from flask import Blueprint, request, jsonify
from datetime import datetime, timedelta
from backend.repository.UserRepository import UserRepository
from backend.utils.EmailService import EmailService, send_email


password_reset_blueprint = Blueprint("password_reset", __name__)
user_repo = UserRepository()
email_service = EmailService()
service = PasswordResetService()

@password_reset_blueprint.route("/forgot-password", methods=["POST"])
def forgot_password():
    data = request.get_json()
    email = data.get("email")

    if not email:
        return jsonify({"error": "Email is required"}), 400

    user = user_repo.get_by_email(email)
    if not user:
        return jsonify({"error": "No account found with that email"}), 404

    reset_code = str(random.randint(1000, 9999))
    expires_at = (datetime.utcnow() + timedelta(minutes=10)).isoformat()
    

    user_repo.save_reset_code(user["email"], reset_code, expires_at)
    subject = "Your Password Reset Code"
    body = (
        f"Hello {user['username']},\n\n"
        f"Your password reset code is: {reset_code}\n\n"
        f"This code expires in 10 minutes.\n\n"
        f"If you didn’t request this, you can ignore this message."
    )
    send_email(email, subject, body)

    return jsonify({"message": "Verification code sent successfully."}), 200


@password_reset_blueprint.route("/reset-password", methods=["POST"])
def reset_password():
    data = request.get_json()
    email = data.get("email")
    new_password = data.get("new_password")
    confirm_password = data.get("confirm_password")

    if not all([email, new_password, confirm_password]):
        return jsonify({"error": "All fields are required"}), 400

    if new_password != confirm_password:
        return jsonify({"error": "Passwords do not match"}), 400

    success = service.update_password(email, new_password)
    if success:
        return jsonify({"message": "Password reset successful"}), 200
    else:
        return jsonify({"error": "Failed to reset password"}), 500



@password_reset_blueprint.route("/verify-code", methods=["POST"])
def verify_code():
    data = request.get_json()
    email = data.get("email")
    code = data.get("code")

    if not email or not code:
        return jsonify({"error": "Email and code are required"}), 400

    verified = service.verify_code(email, code)
    
    if verified:
        user_repo.delete_reset_code(email)
        return jsonify({"message": "Code verified successfully."}), 200
    else:
        return jsonify({"error": "Invalid or expired code."}), 400
