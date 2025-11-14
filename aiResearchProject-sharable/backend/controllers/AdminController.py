# backend/controllers/AdminController.py

from flask import Blueprint, jsonify, request
import sqlite3
import bcrypt

admin_blueprint = Blueprint("admin", __name__)
DB_PATH = "sql.db"


def is_admin(username_or_email: str) -> bool:
    """Check if the requester is an admin."""
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "SELECT role FROM USERS WHERE email = ? OR username = ?",
            (username_or_email, username_or_email),
        )
        row = cur.fetchone()
        return bool(row and row[0] == "admin")


@admin_blueprint.route("/users", methods=["GET"])
def list_users():
    """List all registered users (admin only)."""
    acting = request.args.get("acting")  # username or email of requester
    if not acting or not is_admin(acting):
        return jsonify({"error": "Admin only"}), 403

    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT user_id, username, email, first_name, role, is_active FROM USERS"
        ).fetchall()

    users = [
        {
            "user_id": r[0],
            "username": r[1],
            "email": r[2],
            "first_name": r[3],
            "role": r[4],
            "is_active": bool(r[5]),
        }
        for r in rows
    ]
    return jsonify(users), 200


@admin_blueprint.route("/users/<user_id>/status", methods=["PATCH"])
def update_user_status(user_id):
    """Activate or deactivate a user."""
    data = request.get_json() or {}
    acting = data.get("acting")
    enable = data.get("enable")

    if not acting or not is_admin(acting):
        return jsonify({"error": "Admin only"}), 403

    if enable not in (True, False, 0, 1):
        return jsonify({"error": "Missing or invalid 'enable' field"}), 400

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE USERS SET is_active = ? WHERE user_id = ?",
            (1 if enable else 0, user_id),
        )
        conn.commit()

    return jsonify({"message": "User status updated"}), 200


@admin_blueprint.route("/users/<user_id>", methods=["DELETE"])
def delete_user(user_id):
    """Delete a user account (hard delete)."""
    data = request.get_json() or {}
    acting = data.get("acting")

    if not acting or not is_admin(acting):
        return jsonify({"error": "Admin only"}), 403

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM USERS WHERE user_id = ?", (user_id,))
        conn.commit()

    return jsonify({"message": "User deleted"}), 200


@admin_blueprint.route("/users/<user_id>/password", methods=["PATCH"])
def reset_user_password(user_id):
    """Reset a user's password manually (bcrypt hashed)."""
    data = request.get_json() or {}
    acting = data.get("acting")
    new_password = data.get("new_password")

    if not acting or not is_admin(acting):
        return jsonify({"error": "Admin only"}), 403
    if not new_password:
        return jsonify({"error": "Missing 'new_password'"}), 400

    hashed = bcrypt.hashpw(new_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("UPDATE USERS SET password = ? WHERE user_id = ?", (hashed, user_id))
        conn.commit()

    return jsonify({"message": "Password reset successfully"}), 200


@admin_blueprint.route("/flags/guest_enabled", methods=["GET", "PATCH"])
def manage_guest_flag():
    """Read or update global guest feature flag."""
    if request.method == "GET":
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute(
                "SELECT value FROM FEATURE_FLAGS WHERE key = 'guest_enabled'"
            ).fetchone()
        enabled = row[0] == "1" if row else True
        return jsonify({"guest_enabled": enabled}), 200

    # PATCH request
    data = request.get_json() or {}
    acting = data.get("acting")
    enable = data.get("enable")

    if not acting or not is_admin(acting):
        return jsonify({"error": "Admin only"}), 403
    if enable not in (True, False, 0, 1):
        return jsonify({"error": "Missing or invalid 'enable' field"}), 400

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO FEATURE_FLAGS (key, value)
            VALUES ('guest_enabled', ?)
            ON CONFLICT(key)
            DO UPDATE SET value = excluded.value
            """,
            ("1" if enable else "0",),
        )
        conn.commit()

    return jsonify({"message": "Guest feature flag updated"}), 200


@admin_blueprint.route("/flags/guest_enabled/status", methods=["GET"])
def get_guest_status_for_login():
    """Used by the frontend login page to determine if guest login is allowed."""
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT value FROM FEATURE_FLAGS WHERE key = 'guest_enabled'"
        ).fetchone()

    enabled = row and row[0] == "1"
    return jsonify({"guest_enabled": enabled}), 200


@admin_blueprint.route("/pending_admins", methods=["GET"])
def list_pending_admins():
    """List all pending admin requests (admin only)."""
    acting = request.args.get("acting")
    if not acting or not is_admin(acting):
        return jsonify({"error": "Admin only"}), 403

    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT request_id, username, email, first_name FROM PENDING_ADMINS"
        ).fetchall()

    pending = [
        {
            "request_id": r[0],
            "username": r[1],
            "email": r[2],
            "first_name": r[3],
        }
        for r in rows
    ]
    return jsonify(pending), 200


@admin_blueprint.route("/pending_admins/<int:request_id>/approve", methods=["PATCH"])
def approve_pending_admin(request_id):
    """Approve a pending admin request (admin only)."""
    data = request.get_json() or {}
    acting = data.get("acting")

    if not acting or not is_admin(acting):
        return jsonify({"error": "Admin only"}), 403

    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT username, email, first_name, password FROM PENDING_ADMINS WHERE request_id = ?",
            (request_id,),
        ).fetchone()

        if not row:
            return jsonify({"error": "Request not found"}), 404

        username, email, first_name, password = row

        # Create new admin in USERS
        conn.execute(
            """
            INSERT INTO USERS (user_id, first_name, email, username, password, role, is_active)
            VALUES (hex(randomblob(8)), ?, ?, ?, ?, 'admin', 1)
            """,
            (first_name, email, username, password),
        )

        # Remove from pending list
        conn.execute("DELETE FROM PENDING_ADMINS WHERE request_id = ?", (request_id,))
        conn.commit()

    return jsonify({"message": f"Admin '{username}' approved successfully"}), 200

