from flask import Blueprint, request, jsonify
from backend.models.LoginModel import LoginModel
from backend.services.LoginService import LoginService
import sqlite3

login_blueprint = Blueprint("login", __name__)
login_service = LoginService()


@login_blueprint.route("/auth", methods=['POST'])
def login_request():
    """Authenticate user login with active-status check."""
    data = request.get_json()
    login_model = LoginModel(data.get("username"), data.get("password"))

    # Step 1 — Check if user exists and is active
    with sqlite3.connect("sql.db") as conn:
        cur = conn.execute(
            "SELECT is_active FROM USERS WHERE username = ? OR email = ?",
            (login_model.username, login_model.username)
        )
        row = cur.fetchone()

    # If account exists but is deactivated
    if row and row[0] == 0:
        return jsonify({"error": "Account disabled"}), 403

    # Step 2 — Authenticate credentials
    auth_result = login_service.auth(login_model.username, login_model.password)

    if auth_result == 404:
        return jsonify({"error": "User does not exist"}), 404
    elif auth_result == 401:
        return jsonify({"error": "Invalid password"}), 401
    



    return jsonify({"message": "Login successful"}), 200


@login_blueprint.route("/user_logged", methods=['POST'])
def user_logged():
    """Return user info (id, name, role) after successful login."""
    data = request.get_json()
    username = data.get("username")
    user_id = login_service.get_user_id(username)
    name = login_service.get_first_name(username)

    # Retrieve the user's role from the database
    with sqlite3.connect("sql.db") as conn:
        cur = conn.execute(
            "SELECT role FROM USERS WHERE user_id = ?", (user_id,)
        )
        row = cur.fetchone()
        role = row[0] if row else "user"

    response = {
        "userID": user_id,
        "firstName": name,
        "role": role
    }

    return jsonify(response), 200
