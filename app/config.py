import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory of the project (backend folder)
BASE_DIR = Path(__file__).resolve().parent.parent
env_path = BASE_DIR / ".env"

# Load .env file
load_dotenv(dotenv_path=env_path)

class Config:
    BASE_DIR = BASE_DIR
    GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID")
    GOOGLE_CREDS_JSON = os.getenv("GOOGLE_CREDS_JSON")
    HF_API_TOKEN = os.getenv("HF_API_TOKEN")
    RESEND_API_KEY = os.getenv("RESEND_API_KEY")
    GMAIL_USER = os.getenv("GMAIL_USER")
    GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")
    MATERIALS_BASE_PATH = BASE_DIR / "data" / "materials"
    FAISS_DB_PATH = BASE_DIR / "faiss_db"

config = Config()
