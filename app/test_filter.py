import pandas as pd

def test_filter():
    # Simulate the data from the sheet
    data = [
        {'course_id': 'INF530', 'division': 'A', 'session_date': '2026-03-15', 'files_uploaded': '1.1 The Product Manager.pdf', 'url_count': 0}
    ]
    df = pd.DataFrame(data)
    
    # Simulate values from frontend request
    course_id = "INF530"
    division = "B"
    
    print(f"Testing filter with course_id='{course_id}', division='{division}'")
    
    # Exact logic from main.py
    filtered = df[
        (df['course_id'].astype(str).str.lower() == str(course_id).lower().strip()) &
        (df['division'].astype(str).str.lower() == str(division).lower().strip())
    ]
    
    print(f"Filtered results count: {len(filtered)}")
    if not filtered.empty:
        print("MATCH FOUND (Incorrect behavior!)")
    else:
        print("NO MATCH (Correct behavior)")

    # Test with Div B (as shown in UI)
    division_ui = "Div B"
    print(f"\nTesting filter with course_id='{course_id}', division='{division_ui}'")
    filtered_ui = df[
        (df['course_id'].astype(str).str.lower() == str(course_id).lower().strip()) &
        (df['division'].astype(str).str.lower() == str(division_ui).lower().strip())
    ]
    print(f"Filtered (UI) results count: {len(filtered_ui)}")

if __name__ == "__main__":
    test_filter()
