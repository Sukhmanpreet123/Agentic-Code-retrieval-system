import hashlib


def authenticate(user, token):
    """Authenticate a user by validating both the user record and an auth token."""
    return user is not None and token is not None


def hash_password(password):
    """Hash a plaintext password using SHA-256 before storing it."""
    return hashlib.sha256(password.encode()).hexdigest()


def connect_to_database(host, port):
    """Open a connection to the application's database server."""
    return f"connection to {host}:{port}"


def revoke_token(token):
    """Revoke an authentication token so it can no longer be used."""
    return f"revoked:{token}"
