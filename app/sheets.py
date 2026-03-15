import gspread
import time
from google.oauth2.service_account import Credentials
try:
    from .config import config
except ImportError:
    from config import config

import json

def get_sheets_client():
    """
    Connect to Google Sheets using a service account credentials (JSON string or file path).
    """
    scopes = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]
    
    creds_data = config.GOOGLE_CREDS_JSON
    if not creds_data:
        raise ValueError("GOOGLE_CREDS_JSON not set in environment")
        
    try:
        # Try parsing as JSON string first
        creds_info = json.loads(creds_data)
        creds = Credentials.from_service_account_info(creds_info, scopes=scopes)
    except (json.JSONDecodeError, TypeError):
        # If not valid JSON, treat as file path
        # Resolve relative to BASE_DIR if it's just a filename
        from pathlib import Path
        creds_path = Path(creds_data)
        if not creds_path.is_absolute():
            creds_path = config.BASE_DIR / creds_data
            
        if not creds_path.exists():
            raise FileNotFoundError(f"Credentials file not found at: {creds_path}")
            
        creds = Credentials.from_service_account_file(str(creds_path), scopes=scopes)
        
    client = gspread.authorize(creds)
    return client

def get_worksheet(sheet_id, worksheet_name, headers=None):
    """
    Returns a gspread worksheet object. Creates it if missing if headers are provided.
    Includes retry logic for transient API errors.
    """
    max_retries = 3
    retry_delay = 2 # seconds
    
    for attempt in range(max_retries):
        try:
            client = get_sheets_client()
            sh = client.open_by_key(sheet_id)
            try:
                return sh.worksheet(worksheet_name)
            except gspread.exceptions.WorksheetNotFound:
                if headers:
                    ws = sh.add_worksheet(title=worksheet_name, rows="100", cols="20")
                    ws.append_row(headers)
                    return ws
                raise
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  - Attempt {attempt + 1} failed for '{worksheet_name}': {e}. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay *= 2 # Exponential backoff
            else:
                print(f"  - All {max_retries} attempts failed for '{worksheet_name}': {e}")
                raise

def get_students_for_course_division(course_id: str, division: str):
    """
    Returns a list of students (id, name, email) enrolled in a specific course and division.
    """
    # 1. Get all enrollments
    enrollment_ws = get_worksheet(config.GOOGLE_SHEET_ID, 'Enrollment')
    all_enrollments = enrollment_ws.get_all_records()
    
    # 2. Filter enrollments by course and division
    student_ids = [
        row['student_id'] for row in all_enrollments 
        if str(row['course_id']).strip().upper() == course_id.strip().upper() 
        and str(row['division']).strip().upper() == division.strip().upper()
    ]
    
    if not student_ids:
        return []
        
    # 3. Get all students detail
    students_ws = get_worksheet(config.GOOGLE_SHEET_ID, 'Students')
    all_students = students_ws.get_all_records()
    
    # 4. Filter student details by matching student_ids
    enrolled_students = [
        row for row in all_students 
        if row['student_id'] in student_ids
    ]
    
    
    return enrolled_students

def add_session_history(course_id: str, division: str, session_date: str, filenames: str, url_count: int):
    """
    Appends a new record to the SessionHistory worksheet.
    Expected headers: course_id, division, session_date, files_uploaded, url_count
    """
    try:
        headers = ['course_id', 'division', 'session_date', 'files_uploaded', 'url_count']
        ws = get_worksheet(config.GOOGLE_SHEET_ID, 'SessionHistory', headers=headers)
        row = [course_id.upper(), division.upper(), session_date, filenames, url_count]
        ws.append_row(row)
        try:
            from . import cache
        except ImportError:
            import cache
        cache.refresh_cache("SessionHistory")
        return True
    except Exception as e:
        print(f"Failed to add session history: {e}")
        return False

def get_session_history(course_id: str, division: str):
    """
    Retrieves all session history records for a given course and division.
    Sorted descending by date.
    """
    try:
        headers = ['course_id', 'division', 'session_date', 'files_uploaded', 'url_count']
        ws = get_worksheet(config.GOOGLE_SHEET_ID, 'SessionHistory', headers=headers)
        all_records = ws.get_all_records()
        
        # Filter records
        filtered = [
            row for row in all_records
            if str(row.get('course_id', '')).strip().upper() == course_id.strip().upper()
            and str(row.get('division', '')).strip().upper() == division.strip().upper()
        ]
        
        # Sort descending by date
        filtered.sort(key=lambda x: str(x.get('session_date', '')), reverse=True)
        return filtered
    except Exception as e:
        print(f"Failed to get session history: {e}")
        return []

def remove_session_history(course_id: str, division: str, session_date: str):
    """
    Deletes the row spanning course_id, division, and session_date from SessionHistory.
    """
    try:
        ws = get_worksheet(config.GOOGLE_SHEET_ID, 'SessionHistory')
        all_records = ws.get_all_values()
        
        # Headers are at index 0 (row 1 in gspread)
        headers = [str(x).lower().strip() for x in all_records[0]]
        
        try:
            cid_idx = headers.index('course_id')
            div_idx = headers.index('division')
            date_idx = headers.index('session_date')
        except ValueError:
            print("SessionHistory sheet is missing required headers.")
            return False
            
        # Find the row to delete
        # gspread uses 1-based indexing for rows.
        for idx, row in enumerate(all_records):
            if idx == 0: continue # Skip headers
            
            if (str(row[cid_idx]).strip().upper() == course_id.strip().upper() and
                str(row[div_idx]).strip().upper() == division.strip().upper() and
                str(row[date_idx]).strip() == session_date.strip()):
                
                # We found the row. Delete it (idx + 1 because gspread is 1-indexed)
                ws.delete_rows(idx + 1)
                try:
                    from . import cache
                except ImportError:
                    import cache
                cache.refresh_cache("SessionHistory")
                return True
                
        return False # Row not found
    except Exception as e:
        print(f"Failed to remove session history: {e}")
        return False
