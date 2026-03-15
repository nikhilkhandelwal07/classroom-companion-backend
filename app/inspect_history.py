import sys
sys.path.append('d:/SPJIMR/Course/Term 3/Makers Lab/Group Assignemtn/backend/app')
import sheets
from config import config

def inspect():
    try:
        ws = sheets.get_worksheet(config.GOOGLE_SHEET_ID, 'SessionHistory')
        records = ws.get_all_records()
        print("RECORDS START")
        for r in records:
            print(r)
        print("RECORDS END")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect()
