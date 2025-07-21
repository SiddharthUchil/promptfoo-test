import os
import subprocess
from pathlib import Path

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

# Get API key
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OpenAI API key not found")

print("Running promptfoo evaluation...")

# Run promptfoo with environment variables
env = os.environ.copy()
env['OPENAI_API_KEY'] = api_key

try:
    result = subprocess.run(
        ['npx', 'promptfoo', 'eval', '--config', 'test_config.yaml', '--no-cache'],
        env=env,
        check=True,
        text=True,
        capture_output=True
    )
    print(result.stdout)
except subprocess.CalledProcessError as e:
    print(f"Error running promptfoo: {e}")
    print(f"Output: {e.output}")
    print(f"Error output: {e.stderr}")
except Exception as e:
    print(f"Unexpected error: {e}") 