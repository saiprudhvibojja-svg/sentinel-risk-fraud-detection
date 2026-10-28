import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env if present in workspace
load_dotenv()

def check_kaggle_setup():
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    access_token_file = kaggle_dir / "access_token"
    
    env_user = os.environ.get("KAGGLE_USERNAME")
    env_key = os.environ.get("KAGGLE_KEY")
    env_token = os.environ.get("KAGGLE_API_TOKEN")
    
    print("=" * 50)
    print("Kaggle Environment Diagnostic")
    print("=" * 50)
    print(f"Kaggle folder:      {kaggle_dir} (Exists: {kaggle_dir.exists()})")
    print(f"access_token:       {access_token_file} (Exists: {access_token_file.exists()})")
    print(f"kaggle.json:        {kaggle_json} (Exists: {kaggle_json.exists()})")
    print(f"KAGGLE_USERNAME:    {'Set' if env_user else 'Not set'}")
    print(f"KAGGLE_KEY:         {'Set' if env_key else 'Not set'}")
    print(f"KAGGLE_API_TOKEN:   {'Set' if env_token else 'Not set'}")
    print("-" * 50)
    
    has_creds = (
        access_token_file.exists()
        or kaggle_json.exists()
        or (env_user and env_key)
        or env_token
    )
    
    if not has_creds:
        print("STATUS: FAILED")
        print("Missing credentials! Please provide either:")
        print(f" 1. An access_token file at: {access_token_file}")
        print(f" 2. A kaggle.json file at: {kaggle_json}")
        print(" 3. KAGGLE_USERNAME and KAGGLE_KEY or KAGGLE_API_TOKEN environment variables")
        return False

    try:
        import kaggle
        kaggle.api.authenticate()
        print("STATUS: SUCCESS")
        print(f"Kaggle API authenticated successfully! Logged in as: {kaggle.api.config_values.get('username')}")
        return True
    except Exception as e:
        print("STATUS: AUTHENTICATION ERROR")
        print(f"Error authenticating with Kaggle: {e}")
        return False

if __name__ == "__main__":
    success = check_kaggle_setup()
    sys.exit(0 if success else 1)
