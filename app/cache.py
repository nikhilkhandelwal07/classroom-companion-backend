import pandas as pd
import time
from typing import Dict, Any, Optional
try:
    from .config import config
    from .sheets import get_worksheet, get_sheets_client
except ImportError:
    from config import config
    from sheets import get_worksheet, get_sheets_client

# In-memory storage for DataFrames with timestamps for TTL
# Format: { "SheetName": {"df": DataFrame, "timestamp": float} }
_cache: Dict[str, Dict[str, Any]] = {}

# Cache TTL in seconds (30 seconds)
CACHE_TTL = 30

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
    """Reads all required sheets into memory as pandas DataFrames with timestamps."""
    global _cache
    print("Loading Google Sheets data into cache...")
    
    for sheet_name in REQUIRED_SHEETS:
        load_one_sheet(sheet_name)

def load_one_sheet(sheet_name: str):
    """Loads a single sheet into cache with a current timestamp."""
    try:
        worksheet = get_worksheet(config.GOOGLE_SHEET_ID, sheet_name)
        data = worksheet.get_all_records()
        df = pd.DataFrame(data)
        
        # Normalize column names to lowercase, strip whitespace, and replace spaces with underscores
        df.columns = [c.lower().strip().replace(' ', '_') for c in df.columns]
        
        # Ensure 'present' column is numeric in AttendanceLog
        if sheet_name == "AttendanceLog" and "present" in df.columns:
            df["present"] = pd.to_numeric(df["present"], errors='coerce').fillna(0).astype(int)
        
        _cache[sheet_name] = {
            "df": df,
            "timestamp": time.time()
        }
        print(f"  - Loaded '{sheet_name}' ({len(df)} rows)")
        return df
    except Exception as e:
        print(f"Error loading sheet '{sheet_name}': {str(e)}")
        # Initialize empty DF with columns if possible
        empty_df = pd.DataFrame()
        _cache[sheet_name] = {
            "df": empty_df,
            "timestamp": time.time()
        }
        return empty_df

def get_df(sheet_name: str) -> pd.DataFrame:
    """
    Returns the cached DataFrame for a sheet. 
    Loads on demand if missing or if the cached version has expired (TTL).
    """
    current_time = time.time()
    cached_entry = _cache.get(sheet_name)
    
    # Check if we need to (re)load:
    # 1. Not in cache
    # 2. Cache is empty (might be a previous error)
    # 3. Cache has expired (TTL)
    is_missing = cached_entry is None
    is_empty = cached_entry and cached_entry["df"].empty
    is_expired = cached_entry and (current_time - cached_entry["timestamp"] > CACHE_TTL)
    
    if is_missing or is_empty or is_expired:
        debug_reason = "missing" if is_missing else ("empty" if is_empty else "expired")
        print(f"DEBUG: Reloading '{sheet_name}' from Sheets (reason: {debug_reason})")
        return load_one_sheet(sheet_name)
            
    return cached_entry["df"]

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
