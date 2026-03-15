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
    "Feedback",
    "Participation",
    "SessionHistory",
    "Questions"
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
            
            # Normalize column names to lowercase, strip whitespace, and replace spaces with underscores
            df.columns = [c.lower().strip().replace(' ', '_') for c in df.columns]
            
            # Ensure 'present' column is numeric in AttendanceLog
            if sheet_name == "AttendanceLog" and "present" in df.columns:
                df["present"] = pd.to_numeric(df["present"], errors='coerce').fillna(0).astype(int)
            
            _cache[sheet_name] = df
            print(f"  - Loaded '{sheet_name}' ({len(df)} rows)")
        except Exception as e:
            print(f"  - Error loading '{sheet_name}': {str(e)}")
            # Initialize empty DF with columns if possible
            _cache[sheet_name] = pd.DataFrame()

def load_one_sheet(sheet_name: str):
    """Loads a single sheet into cache."""
    try:
        worksheet = get_worksheet(config.GOOGLE_SHEET_ID, sheet_name)
        data = worksheet.get_all_records()
        df = pd.DataFrame(data)
        df.columns = [c.lower().strip().replace(' ', '_') for c in df.columns]
        _cache[sheet_name] = df
        return df
    except Exception as e:
        print(f"Error loading single sheet '{sheet_name}': {e}")
        return pd.DataFrame()

def get_df(sheet_name: str) -> pd.DataFrame:
    """Returns the cached DataFrame for a sheet. Loads on demand if missing."""
    if sheet_name not in _cache or _cache[sheet_name].empty:
        # Try loading it if it's missing or empty (might have been created recently)
        df = load_one_sheet(sheet_name)
        if not df.empty:
            return df
            
    return _cache.get(sheet_name, pd.DataFrame())

def refresh_cache(sheet_name: str = None):
    """
    Manual trigger to re-load sheets.
    If sheet_name is provided, only that sheet is refreshed.
    Otherwise, all sheets are refreshed.
    """
    if sheet_name:
        print(f"Targeted cache refresh for '{sheet_name}'")
        load_one_sheet(sheet_name)
    else:
        load_all_sheets()
