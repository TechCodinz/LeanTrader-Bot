#!/usr/bin/env python3
"""
Environment Variable Loader
Loads .env file into environment
"""
import os
from pathlib import Path

def load_env():
    """Load environment variables from .env file"""
    env_file = Path(__file__).parent / '.env'
    
    if not env_file.exists():
        print("⚠️  .env file not found")
        return False
    
    loaded = 0
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            # Parse KEY=VALUE
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Set environment variable
                os.environ[key] = value
                loaded += 1
    
    print(f"✅ Loaded {loaded} environment variables")
    return True

if __name__ == "__main__":
    load_env()
    
    # Verify critical keys
    print("\n🔍 Verifying API Keys:")
    
    keys_to_check = [
        'TELEGRAM_BOT_TOKEN',
        'TG_ADMIN_CHAT_ID',
        'BYBIT_API_KEY',
        'BYBIT_SECRET'
    ]
    
    for key in keys_to_check:
        value = os.getenv(key, '')
        if value:
            # Show only first/last 4 chars for security
            if len(value) > 8:
                masked = f"{value[:4]}...{value[-4:]}"
            else:
                masked = "***"
            print(f"  ✅ {key}: {masked}")
        else:
            print(f"  ❌ {key}: Not set")
    
    print("\n✅ Environment loaded successfully!")
