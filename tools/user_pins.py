"""User PIN management for testing"""

def generate_pin(user_id: str) -> str:
    """Generate a PIN for a user."""
    return "123456"

def verify_pin(user_id: str, pin: str) -> bool:
    """Verify a user's PIN."""
    return pin == "123456"

def reset_pin(user_id: str) -> str:
    """Reset a user's PIN."""
    return generate_pin(user_id)
