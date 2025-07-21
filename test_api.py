import os
from pathlib import Path
from openai import OpenAI

def load_env_file():
    """Load environment variables from .env file if it exists."""
    env_file = Path('.env')
    if env_file.exists():
        print("Found .env file")
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
    else:
        print(".env file not found")

# Load .env file
load_env_file()

# Initialize OpenAI client
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OpenAI API key not found")

client = OpenAI(api_key=api_key)

# Test API call
try:
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Say hello!"}],
        temperature=0.1,
        max_tokens=500
    )
    print("\nAPI call successful!")
    print(f"Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"\nAPI call failed: {e}") 