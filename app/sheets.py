import gspread
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

def get_worksheet(sheet_id, worksheet_name):
    """
    Returns a gspread worksheet object.
    """
    client = get_sheets_client()
    sh = client.open_by_key(sheet_id)
    return sh.worksheet(worksheet_name)

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
