import pandas as pd
from typing import Dict, Any
try:
    from .config import config
    from .sheets import get_worksheet, get_sheets_client
except ImportError:
    from config import config
    from sheets import get_worksheet, get_sheets_client

# In-memory storage for DataFrames
# Format: { "SheetName": DataFrame }
_cache: Dict[str, pd.DataFrame] = {}

REQUIRED_SHEETS = [
    "FacultyCourseMap",
    "Students",
    "Enrollment",
    "AttendanceLog",
    "SanctionedLeave",
    "Feedback"
]

def load_all_sheets():
    """Reads all required sheets into memory as pandas DataFrames."""
    global _cache
    print("Loading Google Sheets data into cache...")
    
    for sheet_name in REQUIRED_SHEETS:
        try:
            worksheet = get_worksheet(config.GOOGLE_SHEET_ID, sheet_name)
            data = worksheet.get_all_records()
            df = pd.DataFrame(data)
            
            # Normalize column names to lowercase and strip whitespace
            df.columns = [c.lower().strip() for c in df.columns]
            
            # Ensure 'present' column is numeric in AttendanceLog
            if sheet_name == "AttendanceLog" and "present" in df.columns:
                df["present"] = pd.to_numeric(df["present"], errors='coerce').fillna(0).astype(int)
            
            _cache[sheet_name] = df
            print(f"  - Loaded '{sheet_name}' ({len(df)} rows)")
        except Exception as e:
            print(f"  - Error loading '{sheet_name}': {str(e)}")
            # Initialize empty DF to avoid crashes
            _cache[sheet_name] = pd.DataFrame()

def get_df(sheet_name: str) -> pd.DataFrame:
    """Returns the cached DataFrame for a sheet."""
    return _cache.get(sheet_name, pd.DataFrame())

def refresh_cache():
    """Manual trigger to re-load all sheets."""
    load_all_sheets()
