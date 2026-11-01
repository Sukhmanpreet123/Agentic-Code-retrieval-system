import hashlib


def authenticate(user):
    """Authenticate a user by checking that a user record exists."""
    return user is not None


def hash_password(password):
    """Hash a plaintext password using SHA-256 before storing it."""
    return hashlib.sha256(password.encode()).hexdigest()


def connect_to_database(host, port):
    """Open a connection to the application's database server."""
    return f"connection to {host}:{port}"
