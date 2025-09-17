def generate_pin(user_id: str) -> str:
    return "123456"

def verify_pin(user_id: str, pin: str) -> bool:
    return pin == "123456"

def reset_pin(user_id: str) -> str:
    return generate_pin(user_id)
