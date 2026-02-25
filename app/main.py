import uuid
import re
import pandas as pd
import json
import time
import os
import tempfile
import smtplib
from typing import Dict, Optional, Annotated, List, Any
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import requests
from fastapi import FastAPI, Depends, HTTPException, Header, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.openapi.utils import get_openapi
# RAG Imports and Shared Embeddings
try:
    from .rag import embeddings
except ImportError:
    from rag import embeddings

try:
    from .config import config
    from .sheets import get_sheets_client, get_worksheet, get_students_for_course_division
    from . import cache
    from . import rag
except ImportError:
    from config import config
    from sheets import get_sheets_client, get_worksheet, get_students_for_course_division
    import cache
    import rag

# Set up email delivery logger
import logging
email_logger = logging.getLogger("email_delivery")
email_logger.setLevel(logging.INFO)
log_file = os.path.join(config.BASE_DIR, "email_delivery.log")
handler = logging.FileHandler(log_file)
handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
email_logger.addHandler(handler)

app = FastAPI(title="Classroom Companion API v2.3")

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version="2.3",
        openapi_version="3.0.3",
        description=app.description,
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

HF_API_URL = "https://router.huggingface.co/hf-inference/models/cardiffnlp/twitter-roberta-base-sentiment-latest"
MISTRAL_URL = "https://router.huggingface.co/v1/chat/completions"
# Using the specific HF router URL scheme as requested.

@app.on_event("startup")
async def startup_event():
    # Load all sheets into memory at startup
    cache.load_all_sheets()

class EmailRequest(BaseModel):
    course_id: str
    division: str
    threshold: float = 75.0

class MailMaterialRequest(BaseModel):
    course_id: str
    divisions: List[str]
    subject: str
    message: str
    summary: Optional[Dict] = None
    filenames: Optional[List[str]] = []
    urls: Optional[List[str]] = []


# In-memory session store: {token: faculty_email}
sessions: Dict[str, str] = {}

class LoginRequest(BaseModel):
    email: str
    password: str

# Enable CORS for frontend development and production
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        'http://localhost:5173',
        'https://classroom-companion-spjimr.vercel.app'
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def verify_token(authorization: Optional[str] = Header(None)):
    if not authorization:
        print(f"DEBUG: Missing Authorization header")
        raise HTTPException(status_code=401, detail="Missing authorization header")
    
    if not authorization.startswith("Bearer "):
        print(f"DEBUG: Invalid header format: {authorization}")
        raise HTTPException(status_code=401, detail="Invalid authorization header format")
    
    token = authorization.split(" ")[1]
    if token not in sessions:
        print(f"DEBUG: Token not found in sessions. Token: {token}. Valid sessions: {list(sessions.keys())}")
        raise HTTPException(status_code=401, detail="Invalid or expired token")
        
    return sessions[token]

def extract_json_safe(content: str):
    """Robustly extracts and cleans JSON from AI response strings."""
    try:
        # 1. Basic Cleaning
        # Remove markdown code blocks if present
        cleaned = re.sub(r'```(?:json)?\s*(.*?)\s*```', r'\1', content, flags=re.DOTALL)
        cleaned = cleaned.strip()
        
        # 2. Scope extraction
        start = cleaned.find('{')
        end = cleaned.rfind('}') + 1
        
        if start == -1 or end <= start:
            # Fallback for when AI returns direct JSON without braces (rare)
            try:
                return json.loads(cleaned)
            except:
                raise ValueError("No JSON-like structure found")

        json_str = cleaned[start:end]
        
        # 3. Aggressive control character removal
        # AI often puts literal newlines inside string values which breaks json.loads
        # We replace them with a space ONLY if they are not preceded by a backslash
        json_str = re.sub(r'(?<!\\)\n', ' ', json_str)
        json_str = re.sub(r'(?<!\\)\t', ' ', json_str)
        
        # 4. Try parsing first
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as first_e:
            # 5. HEURISTICS FOR COMMON AI ERRORS
            fixed = json_str
            
            # (A) Fix missing commas between key-value pairs or list items
            # Example: "summary": "text" "working_well": [...] -> missing comma
            # Look for " followed by whitespace then " that isn't preceded by a colon
            # This is tricky, but let's try a safer version:
            # If a line seems to end a value and the next starts a key/value
            fixed = re.sub(r'(\"|\]|\})\s*\"(\w+)\"\s*:', r'\1, "\2":', fixed)
            
            # (B) Fix missing commas in arrays of strings
            # Example: ["point 1" "point 2"] -> missing comma
            fixed = re.sub(r'\"(\s+)\"', r'", "', fixed)
            
            # (C) Remove markdown bolding prefixes that end up outside quotes like **"
            fixed = re.sub(r'\*\*\s*"', '"', fixed)
            
            # (D) Remove trailing commas before } or ]
            fixed = re.sub(r',\s*([}\]])', r'\1', fixed)

            # (E) Fix unescaped quotes inside string values (Best effort)
            # Find patterns like "key": "value with "inner" quotes"
            # This is hard to regex perfectly, but we can try to escape quotes
            # that are not at the start/end of a JSON structure.
            # (Skip for now as it's very prone to breaking valid structures)

            try:
                return json.loads(fixed)
            except json.JSONDecodeError as second_e:
                print(f"DEBUG: JSON Fix Failed. Original Error: {first_e.msg}. Fixed Error: {second_e.msg}")
                # Log a bit more around the error
                snippet = fixed[max(0, second_e.pos-100):min(len(fixed), second_e.pos+100)]
                print(f"DEBUG: Problematic fixed snippet: {snippet}")
                raise second_e

    except Exception as e:
        print(f"JSON extraction failed: {e}")
        # Log the raw content to a file for investigation
        log_path = os.path.join(config.BASE_DIR, "json_fail.log")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n--- {time.ctime()} ---\n")
            f.write(f"Error: {str(e)}\n")
            f.write(f"Content: {content}\n")
        raise ValueError(f"No valid JSON found in response: {str(e)}")

class UrlRequest(BaseModel):
    url: str
    course_id: str
    division: str

class GenerateRequest(BaseModel):
    course_id: str
    division: str

class SessionPlanRequest(BaseModel):
    course_id: str
    division: str
    session_duration: int = 70

class ChatRequest(BaseModel):
    question: str
    course_id: str
    division: str
    history: List[Dict[str, Any]] = []

class ClearRequest(BaseModel):
    course_id: str
    division: str

@app.post("/refresh-cache")
async def refresh_all_cache(faculty_email: str = Depends(verify_token)):
    try:
        cache.refresh_cache()
        return {"success": True, "message": "Cache refreshed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/login")
async def login(request: LoginRequest):
    try:
        # Read from cache
        df_faculty = cache.get_df("FacultyCourseMap")
        if df_faculty.empty:
            return {"success": False, "message": "Faculty data not loaded. Please refresh cache."}
            
        # Check for matching credentials
        faculty_courses = []
        user_found = False
        
        email_clean = request.email.strip().lower()
        
        for _, record in df_faculty.iterrows():
            record_email = str(record.get("faculty_email", "")).strip().lower()
            record_password = str(record.get("password", "")).strip()
            
            if record_email == email_clean and record_password == request.password:
                user_found = True
                faculty_courses.append({
                    "course_id": record.get("course_id"),
                    "course_name": record.get("course_name"),
                    "division": record.get("division")
                })
        
        if user_found:
            token = str(uuid.uuid4())
            sessions[token] = email_clean
            
            return {
                "success": True,
                "token": token,
                "faculty_email": email_clean,
                "courses": faculty_courses
            }
        else:
            return {"success": False, "message": "Invalid credentials"}
            
    except Exception as e:
        return {"success": False, "message": f"Login error: {str(e)}"}

@app.get("/test-sheets")
async def test_sheets():
    try:
        client = get_sheets_client()
        # Just try to get the spreadsheets to verify auth
        client.list_spreadsheet_files()
        return {"status": "success", "message": "Successfully connected to Google Sheets"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/email-material")
async def email_material(request: MailMaterialRequest, faculty_email: str = Depends(verify_token)):
    sent_count = 0
    errors = []
    
    # Imports for attachments
    from email.mime.application import MIMEApplication
    
    email_logger.info(f"START: Email Material Share for Course {request.course_id} across divisions {request.divisions}")
    
    for div in request.divisions:
        try:
            # 1. Get students for this division
            students = get_students_for_course_division(request.course_id, div)
            if not students:
                msg = f"No students found for Course {request.course_id} Division {div}"
                print(msg)
                email_logger.info(msg)
                continue
            
            # 2. Attachments Preparation
            attachments = []
            course_div_path = config.MATERIALS_BASE_PATH / f"{request.course_id}_{div}"
            if request.filenames and course_div_path.exists():
                for fname in request.filenames:
                    file_path = course_div_path / fname
                    if file_path.exists():
                        try:
                            with open(file_path, "rb") as f:
                                data = f.read()
                                attachments.append((fname, data))
                        except Exception as ef:
                            print(f"Failed to read attachment {fname}: {ef}")

            # Prepare Content
            url_list_html = ""
            if request.urls:
                url_list_html = "<h4>Reference URLs:</h4><ul>" + "".join([f'<li><a href="{u}">{u}</a></li>' for u in request.urls]) + "</ul>"

            summary_html = ""
            if request.summary:
                s_data = request.summary
                s_points = s_data.get('summary', [])
                if isinstance(s_points, list) and s_points:
                    summary_html = "<h3>Session Summary:</h3><ul>" + "".join([f"<li>{p}</li>" for p in s_points]) + "</ul>"
            
            # 3. Iterate and send to each student
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(config.GMAIL_USER, config.GMAIL_APP_PASSWORD)
                
                for s in students:
                    student_email = s.get('email', '').strip()
                    if not student_email or "@" not in student_email:
                        email_logger.warning(f"INVALID EMAIL for Student {s.get('student_id')} ({s.get('student_name')}): {student_email}")
                        continue
                        
                    try:
                        msg = MIMEMultipart()
                        msg['From'] = f"Classroom Companion <{config.GMAIL_USER}>"
                        msg['To'] = student_email
                        # Use subject from request, fallback to generic if empty
                        msg['Subject'] = request.subject if request.subject else f"Session Material - {request.course_id}"
                        
                        body = f"""
                        <html>
                        <body style="font-family: sans-serif; line-height: 1.6; color: #333;">
                            <div style="max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #eee; border-radius: 10px;">
                                <h2 style="color: #1e3a8a; border-bottom: 2px solid #38bdf8; padding-bottom: 10px;">Session Material Share</h2>
                                <p>Dear {s.get('student_name', 'Student')},</p>
                                <p>{request.message}</p>
                                
                                {summary_html}
                                {url_list_html}

                                <p style="font-size: 0.8em; color: #64748b; margin-top: 30px; border-top: 1px solid #eee; padding-top: 10px;">
                                    This material was shared by your faculty. (System Version: 2.4 | NO-BCC Active)
                                </p>
                            </div>
                        </body>
                        </html>
                        """
                        msg.attach(MIMEText(body, 'html'))
                        
                        # Attach the PDFs
                        for fname, data in attachments:
                            part = MIMEApplication(data, Name=fname)
                            part['Content-Disposition'] = f'attachment; filename="{fname}"'
                            msg.attach(part)
                        
                        # Send directly to student ONLY - NO BCC parameter used
                        server.send_message(msg)
                        
                        sent_count += 1
                        email_logger.info(f"SENT: {student_email} (Course: {request.course_id}, Div: {div})")
                    except Exception as ese:
                        email_logger.error(f"FAIL: {student_email} - {str(ese)}")
            
        except Exception as e:
            print(f"Failed to process division {div}: {str(e)}")
            errors.append(f"Division {div}: {str(e)}")
            email_logger.error(f"DIV ERROR: {div} - {str(e)}")
            
    email_logger.info(f"COMPLETED: Sent {sent_count} emails. (User requested: BCC removed, Summary only)")
    
    if sent_count == 0 and request.divisions:
        raise HTTPException(status_code=500, detail=f"Failed to send any emails. Errors: {'; '.join(errors)}")
        
    return {"sent": sent_count, "errors": errors}

@app.get("/attendance")
async def get_attendance(
    course_id: str, 
    division: str, 
    faculty_email: str = Depends(verify_token)
):
    try:
        # 1. Get DataFrames from cache
        df_log = cache.get_df("AttendanceLog")
        df_students = cache.get_df("Students")
        df_enrollment = cache.get_df("Enrollment")
        df_leaves = cache.get_df("SanctionedLeave")

        if df_log.empty or df_students.empty or df_enrollment.empty:
             raise HTTPException(status_code=500, detail="Sheet data not loaded. Please ensure tabs exist and refresh cache.")

        # 2. Filter data
        # Filter log by course
        course_log = df_log[df_log['course_id'] == course_id]

        # Enrolled students for this course and division
        enrolled = df_enrollment[
            (df_enrollment['course_id'] == course_id) & 
            (df_enrollment['division'] == str(division))
        ]
        
        if enrolled.empty:
            return {
                "summary": {
                    "total_students": 0,
                    "avg_attendance": 0,
                    "count_good": 0,
                    "count_warning": 0,
                    "count_poor": 0
                },
                "students": []
            }

        # Get student details (names and emails)
        enrolled_students = pd.merge(
            enrolled, 
            df_students[['student_id', 'student_name', 'email']], 
            on='student_id', 
            how='left'
        )

        results = []
        for _, student in enrolled_students.iterrows():
            sid = student['student_id']
            # Filter log for THIS specific student in THIS course
            student_df = course_log[course_log['student_id'] == sid]
            
            # (2) count total number of rows for that student — this is total_sessions
            total_s = len(student_df)
            
            # (3) sum the 'present' column converting to int first — this is sessions_attended
            if 'present' in student_df.columns and total_s > 0:
                attended_s = int(student_df['present'].astype(int).sum())
            else:
                attended_s = 0
            
            # (4) calculate percentage as (sessions_attended / total_sessions) * 100
            percentage = (attended_s / total_s * 100) if total_s > 0 else 0
            
            # Determine status
            if percentage >= 85:
                status = "Good"
            elif percentage >= 65:
                status = "Warning"
            else:
                status = "Poor"
            
            # Check for leaves
            student_leave = df_leaves[
                (df_leaves['student_id'] == sid) & 
                (df_leaves['course_id'] == course_id)
            ]
            sanctioned_leave = None
            if not student_leave.index.empty:
                leave_row = student_leave.iloc[0]
                sanctioned_leave = {
                    "leave_type": leave_row['leave_type'],
                    "date": str(leave_row['date'])
                }

            results.append({
                "student_id": sid,
                "student_name": student['student_name'],
                "sessions_attended": attended_s,
                "total_sessions": total_s,
                "percentage": round(percentage, 1),
                "status": status,
                "sanctioned_leave": sanctioned_leave
            })

        # Calculate summary stats
        total_count = len(results)
        avg_att = sum(s['percentage'] for s in results) / total_count if total_count > 0 else 0
        count_good = sum(1 for s in results if s['status'] == "Good")
        count_warning = sum(1 for s in results if s['status'] == "Warning")
        count_poor = sum(1 for s in results if s['status'] == "Poor")

        return {
            "summary": {
                "total_students": total_count,
                "avg_attendance": round(avg_att, 1),
                "count_good": count_good,
                "count_warning": count_warning,
                "count_poor": count_poor
            },
            "students": results
        }

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/email-attendance")
async def email_attendance(request: EmailRequest, faculty_email: str = Depends(verify_token)):
    try:
        # 1. Get attendance data (reusing same logic as /attendance)
        df_log = cache.get_df("AttendanceLog")
        df_students = cache.get_df("Students")
        df_enrollment = cache.get_df("Enrollment")
        df_faculty = cache.get_df("FacultyCourseMap")

        if df_log.empty or df_students.empty or df_enrollment.empty:
             raise HTTPException(status_code=500, detail="Sheet data not loaded.")

        # Get course name
        course_name = "Course"
        course_info = df_faculty[df_faculty['course_id'] == request.course_id]
        if not course_info.empty:
            course_name = course_info.iloc[0]['course_name']

        # Filter students
        enrolled = df_enrollment[
            (df_enrollment['course_id'] == request.course_id) & 
            (df_enrollment['division'] == str(request.division))
        ]
        
        enrolled_students = pd.merge(
            enrolled, 
            df_students[['student_id', 'student_name', 'email']], 
            on='student_id', 
            how='left'
        )

        course_log = df_log[df_log['course_id'] == request.course_id]
        
        sent_count = 0
        notified_students = []

        for _, student in enrolled_students.iterrows():
            sid = student['student_id']
            sname = student['student_name']
            student_df = course_log[course_log['student_id'] == sid]
            total_s = len(student_df)
            
            if 'present' in student_df.columns and total_s > 0:
                attended_s = int(student_df['present'].astype(int).sum())
            else:
                attended_s = 0
            
            percentage = (attended_s / total_s * 100) if total_s > 0 else 0
            
            if percentage < request.threshold:
                # Get student email
                target_email = str(student.get('email', '')).strip()
                if not target_email or "@" not in target_email:
                    target_email = faculty_email # Fallback to faculty if invalid
                
                # Send email using Gmail SMTP
                try:
                    msg = MIMEMultipart()
                    msg['From'] = f"Classroom Companion <{config.GMAIL_USER}>"
                    msg['To'] = target_email
                    msg['Subject'] = f"Attendance Warning - {course_name}"
                    
                    body = f"""
                        <div style="font-family: sans-serif; padding: 20px; border: 1px solid #e2e8f0; border-radius: 12px;">
                            <h2 style="color: #1e3a8a;">Attendance Warning</h2>
                            <p>Dear {sname},</p>
                            <p>Your current attendance in <strong>{course_name}</strong> is <strong>{round(percentage, 1)}%</strong>, which is below the required {request.threshold}%.</p>
                            <p>Please ensure regular attendance to avoid any academic consequences.</p>
                            <p style="margin-top: 20px; color: #64748b;">Regards,<br>Faculty Team</p>
                        </div>
                    """
                    msg.attach(MIMEText(body, 'html'))
                    
                    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                        server.login(config.GMAIL_USER, config.GMAIL_APP_PASSWORD)
                        server.send_message(msg)
                        
                    sent_count += 1
                    notified_students.append({"name": sname, "percentage": round(percentage, 1)})
                except Exception as e:
                    print(f"Failed to send email for {sname}: {str(e)}")

        return {
            "success": True,
            "sent": sent_count,
            "students": notified_students
        }

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/feedback")
async def get_feedback(course_id: str, division: str, faculty_email: str = Depends(verify_token)):
    try:
        import time
        import json
        
        # 1. Get Feedback data from cache
        df_feedback = cache.get_df("Feedback")
        
        empty_res = {
            "total_responses": 0,
            "avg_rating": 0,
            "positive_count": 0,
            "neutral_count": 0,
            "negative_count": 0,
            "positive_pct": 0,
            "negative_pct": 0,
            "rating_distribution": [
                {"stars": 5, "count": 0},
                {"stars": 4, "count": 0},
                {"stars": 3, "count": 0},
                {"stars": 2, "count": 0},
                {"stars": 1, "count": 0}
            ],
            "avg_rating_pct": 0,
            "ai_analysis": {
                "summary": "No feedback data available for this session.",
                "working_well": [],
                "areas_to_improve": []
            }
        }

        if df_feedback.empty:
            return empty_res
        
        # 2. Filter by course and division
        filtered = df_feedback[
            (df_feedback['course_id'] == course_id) & 
            (df_feedback['division'] == str(division))
        ]
        
        if filtered.empty:
            return empty_res

        # 3. Process sentiments using HF Inference API with Retries
        headers = {"Authorization": f"Bearer {config.HF_API_TOKEN}"}
        
        comments_list = []
        pos_c = 0
        neu_c = 0
        neg_c = 0
        total_rating = 0
        rating_counts = {5: 0, 4: 0, 3: 0, 2: 0, 1: 0}
        
        def get_sentiment_with_retry(text, max_retries=3):
            for i in range(max_retries):
                try:
                    response = requests.post(HF_API_URL, headers=headers, json={"inputs": text}, timeout=15)
                    results = response.json()
                    
                    if isinstance(results, dict) and "error" in results:
                         if "loading" in results.get("error", "").lower():
                             print(f"Model loading, retry {i+1} in 10s...")
                             time.sleep(10)
                             continue
                    
                    # HF returns [[{"label": "LABEL_2", "score": 0.9}, ...]]
                    # HF returns [[{"label": "LABEL_2", "score": 0.9}, ...]]
                    if isinstance(results, list) and len(results) > 0:
                        inner = results[0]
                        if isinstance(inner, list) and len(inner) > 0:
                            sorted_results = sorted(inner, key=lambda x: x.get('score', 0), reverse=True)
                            top = sorted_results[0]
                            return str(top.get('label', '')).lower()
                    return "neutral"
                except Exception as e:
                    print(f"Sentiment retry {i+1} failed: {e}")
                    time.sleep(2)
            return "Neutral"

        for _, row in filtered.iterrows():
            text = str(row.get('comment', row.get('text', ''))).strip()
            rating = 0
            try:
                val = row.get('rating', 0)
                rating = int(float(val)) if val else 0
                if 1 <= rating <= 5:
                    rating_counts[rating] += 1
                rating = float(val) if val else 0
            except:
                rating = 0
            
            total_rating += rating
            sentiment_label = "Neutral"
            if text and config.HF_API_TOKEN:
                raw_label = get_sentiment_with_retry(text)
                # Apply hybrid logic:
                # if rating is 4 or 5 AND model says Positive or Neutral → Positive. 
                # If rating is 3 OR (rating 4-5 AND model says Negative) → Neutral. 
                # If rating is 1 or 2 → Negative regardless of model.
                if rating >= 4:
                    if "pos" in raw_label or "neu" in raw_label:
                        sentiment_label = "Positive"
                    else:
                        sentiment_label = "Neutral"
                elif rating == 3:
                    sentiment_label = "Neutral"
                else: # rating 1 or 2
                    sentiment_label = "Negative"
            
            if sentiment_label == "Positive": pos_c += 1
            elif sentiment_label == "Negative": neg_c += 1
            else: neu_c += 1

            comments_list.append({
                "text": text,
                "sentiment": sentiment_label,
                "rating": rating
            })
            
        total = len(filtered)
        avg_rating = total_rating / total if total > 0 else 0
        
        # 4. AI Analysis using Mistral-7B
        ai_analysis = {
            "summary": "AI could not generate a summary at this time.",
            "working_well": [],
            "areas_to_improve": []
        }
        
        if comments_list and config.HF_API_TOKEN:
            try:
                # Pass all comments AND their ratings to Mistral: 'Rating X/5: [comment text]'
                formatted_comments = [f"Rating {int(c['rating'])}/5: {c['text']}" for c in comments_list if c['text']]
                all_text = "\n- ".join(formatted_comments)
                
                prompt = f"""You are a concise academic feedback analyst. Analyze the following student feedback and respond ONLY with a JSON object — no extra text, no markdown, no code blocks.

Rules:
- Summary must be exactly 4-5 sentences maximum
- Do not quote or restate individual comments
- Identify PATTERNS across multiple comments, not individual opinions
- Be direct and specific — no filler phrases like 'overall' or 'in general'
- working_well and areas_to_improve must each have exactly 3 bullet points
- Each bullet point is one clear sentence maximum

Feedback (Rating/5: Comment):
{all_text}

Respond with exactly this JSON structure (ensure all keys and values are in double quotes, and all items are comma-separated):
{{"summary": "3-4 sentence synthesis here", "working_well": ["point 1", "point 2", "point 3"], "areas_to_improve": ["point 1", "point 2", "point 3"]}}"""
                
                # Use OpenAI-compatible chat payload for Mistral on router.huggingface.co
                payload = {
                    "model": "mistralai/Mistral-7B-Instruct-v0.2",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 600,
                    "temperature": 0.3
                }
                
                # Retry mechanism for Mistral as well
                for i in range(3):
                    res = requests.post(MISTRAL_URL, headers=headers, json=payload, timeout=30)
                    ai_results = res.json()
                    
                    # Log to file for remote debugging
                    log_path = os.path.join(config.BASE_DIR, "mistral_debug.log")
                    with open(log_path, "a") as f:
                        f.write(f"\n--- {time.ctime()} ---\n")
                        f.write(f"Status: {res.status_code}\n")
                        f.write(f"Response: {json.dumps(ai_results)[:1000]}\n")
                    
                    if isinstance(ai_results, dict) and "error" in ai_results:
                        if "loading" in ai_results.get("error", "").lower():
                            print(f"Mistral loading, retry {i+1} in 10s...")
                            time.sleep(10)
                            continue
                    
                    # OpenAI style parsing: results['choices'][0]['message']['content']
                    if isinstance(ai_results, dict) and "choices" in ai_results:
                        try:
                            content = ai_results['choices'][0]['message']['content']
                            ai_analysis = extract_json_safe(content)
                            if all(k in ai_analysis for k in ["summary", "working_well", "areas_to_improve"]):
                                break
                        except Exception as e:
                            print(f"Mistral JSON parse error in feedback: {e}")
                            log_path = os.path.join(config.BASE_DIR, "mistral_debug.log")
                            with open(log_path, "a") as f:
                                f.write(f"Parse Error: {str(e)}\nRaw Content: {content}\n")
                    
                    time.sleep(10) # Wait before retry if not successful
                    continue
            except Exception as e:
                print(f"AI Analysis error: {e}")

        return {
            "total_responses": total,
            "avg_rating": round(avg_rating, 1),
            "positive_count": pos_c,
            "neutral_count": neu_c,
            "negative_count": neg_c,
            "positive_pct": round((pos_c / total) * 100, 1) if total > 0 else 0,
            "neutral_pct": round((neu_c / total) * 100, 1) if total > 0 else 0,
            "negative_pct": round((neg_c / total) * 100, 1) if total > 0 else 0,
            "avg_rating_pct": round((avg_rating / 5) * 100, 1) if total > 0 else 0,
            "rating_distribution": [
                {"stars": s, "count": rating_counts[s]} for s in [5,4,3,2,1]
            ],
            "comments": comments_list,
            "ai_analysis": ai_analysis
        }

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

def call_mistral_inference(prompt: str, max_tokens: int = 600, temperature: float = 0.3):
    """Calls Mistral-7B-Instruct-v0.2 via HF Inference API Router."""
    headers = {"Authorization": f"Bearer {config.HF_API_TOKEN}"}
    payload = {
        "model": "mistralai/Mistral-7B-Instruct-v0.2",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    for i in range(3):
        try:
            res = requests.post(MISTRAL_URL, headers=headers, json=payload, timeout=45)
            
            # Debugging logs
            print(f"DEBUG: Mistral Status: {res.status_code}")
            if res.status_code != 200:
                print(f"DEBUG: Mistral Response Text: {res.text[:500]}")
            
            if not res.text.strip():
                print("DEBUG: Mistral returned empty response")
                time.sleep(5)
                continue

            try:
                ai_results = res.json()
            except Exception as e:
                print(f"DEBUG: Failed to parse JSON. Error: {e}")
                print(f"DEBUG: Raw response: {res.text[:200]}")
                time.sleep(5)
                continue
            
            if isinstance(ai_results, dict) and "choices" in ai_results:
                return ai_results['choices'][0]['message']['content']
            
            if isinstance(ai_results, dict) and "error" in ai_results:
                if "loading" in ai_results.get("error", "").lower():
                    print(f"Mistral loading, retry {i+1} in 10s...")
                    time.sleep(10)
                    continue
            break
        except Exception as e:
            print(f"Mistral call error: {e}")
            time.sleep(2)
    return ""

@app.post("/upload-material")
async def upload_material(
    files: List[UploadFile] = File(...),
    course_id: str = Form(...),
    division: str = Form(...),
    faculty_email: str = Depends(verify_token)
):
    total_chunks = 0
    # Create course-specific materials directory
    course_materials_path = config.MATERIALS_BASE_PATH / f"{course_id}_{division}"
    course_materials_path.mkdir(parents=True, exist_ok=True)
    
    for file in files:
        contents = await file.read()
        target_path = course_materials_path / file.filename
        
        # Save original file persistently for email attachments
        with open(target_path, "wb") as f:
            f.write(contents)
            
        try:
            # Delegate to rag.py for all processing and indexing
            chunks_added = rag.process_pdf(str(target_path), course_id, division)
            total_chunks += chunks_added
        except Exception as e:
            print(f"Error processing {file.filename}: {e}")
            if target_path.exists():
                target_path.unlink()
            raise HTTPException(status_code=500, detail=f"Failed to process {file.filename}: {str(e)}")
        
    return {
        'status': 'success',
        'chunks_added': total_chunks,
        'files_processed': len(files)
    }

@app.post("/add-url")
async def add_url(request: UrlRequest, faculty_email: str = Depends(verify_token)):
    try:
        chunks = rag.process_url(request.url, request.course_id, request.division)
        return {"success": True, "message": f"URL processed, added {chunks} chunks."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-summary")
async def generate_summary(request: GenerateRequest, faculty_email: str = Depends(verify_token)):
    context = rag.get_context(request.course_id, request.division, query="Executive summary, key concepts, and main topics of the course material", n_results=10)
    if not context:
        raise HTTPException(status_code=404, detail="No material found for this course/division. Please upload documents or add URLs first.")
        
    print(f"DEBUG: Context for summary (first 200 chars): {str(context)[:200]}")
    
    prompt = f"""You are an expert academic assistant. Analyze the provided course material and respond with a strictly formatted JSON object.

Material:
\"\"\"
{context}
\"\"\"

Task:
1. Provide a 5-point session summary.
2. Identify 5 key concepts with a one-line explanation for each.
3. Generate 4 discussion prompts for classroom use.

Constraint:
- Respond ONLY with a valid JSON object.
- DO NOT use markdown bolding (**) outside of double quotes.
- Ensure all double quotes inside string values are escaped with a backslash (\").
- Ensure all keys and values are comma-separated correctly.
- No conversational text, no markdown code blocks, no preamble.

Required JSON Structure:
{{
  "summary": ["point 1", "point 2", "point 3", "point 4", "point 5"],
  "key_concepts": [
    {{"concept": "name", "explanation": "description"}},
    {{"concept": "name", "explanation": "description"}},
    {{"concept": "name", "explanation": "description"}},
    {{"concept": "name", "explanation": "description"}},
    {{"concept": "name", "explanation": "description"}}
  ],
  "discussion_prompts": ["prompt 1", "prompt 2", "prompt 3", "prompt 4"]
}}"""
    
    content = call_mistral_inference(prompt, max_tokens=1500)
    if content:
        try:
            return extract_json_safe(content)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"AI generated invalid JSON: {str(e)}")
    raise HTTPException(status_code=500, detail="Failed to generate summary")

@app.post("/generate-session-plan")
async def generate_session_plan(request: SessionPlanRequest, faculty_email: str = Depends(verify_token)):
    context = rag.get_context(request.course_id, request.division, query="Detailed session plan structure, activities, and timing", n_results=15)
    if not context:
        raise HTTPException(status_code=404, detail="No material found for this course/division. Please upload documents or add URLs first.")
        
    print(f"DEBUG: Context for session plan (first 200 chars): {str(context)[:200]}")
    
    prompt = f"""Create a detailed {request.session_duration}-minute session plan for a business school class based on the provided material.

Material:
\"\"\"
{context}
\"\"\"

Structure the plan into these time blocks:
- warm_up (10 min)
- concept_intro (20 min)
- case_discussion (20 min)
- group_activity (20 min)
- debrief (15 min)
- wrap_up (5 min)

Required JSON Structure:
{{
  "session_title": "string",
  "blocks": [
    {{
      "type": "block_name",
      "duration": "X min",
      "title": "Block Title",
      "activity": "Detailed description",
      "questions": ["q1", "q2"]
    }},
    ...
  ]
}}

Constraint: 
- Respond ONLY with a valid JSON object.
- DO NOT use markdown bolding (**) outside of double quotes.
- Ensure all double quotes inside string values are escaped with a backslash (\").
- Ensure all keys and values are comma-separated correctly.
- No conversational text, no markdown code blocks, no preamble.
"""
    
    content = call_mistral_inference(prompt, max_tokens=2000)
    if content:
        try:
            return extract_json_safe(content)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"AI generated invalid JSON: {str(e)}")
    raise HTTPException(status_code=500, detail="Failed to generate session plan")

@app.post("/chat")
async def chat_with_material(request: ChatRequest, faculty_email: str = Depends(verify_token)):
    context = rag.get_context(request.course_id, request.division, query=request.question, n_results=5)
    if not context:
        context = "No relevant material found."
        
    history_str = ""
    for msg in request.history[-6:]: # Last 3 turns
        role = "Faculty" if msg.get("role") == "faculty" else "Assistant"
        history_str += f"{role}: {msg.get('content')}\n"

    prompt = f"""You are a teaching assistant. Answer the faculty question based on the provided material and conversation history.

Material:
\"\"\"
{context}
\"\"\"

History:
{history_str}

Question: {request.question}
Answer:"""
    
    content = call_mistral_inference(prompt)
    return {"answer": content if content else "I'm sorry, I couldn't find an answer in the provided material."}

@app.post("/clear-material")
@app.delete("/clear-material")
async def clear_material(request: ClearRequest, faculty_email: str = Depends(verify_token)):
    try:
        rag.clear_session_material(request.course_id, request.division)
        return {"success": True, "message": "Material cleared successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list-materials")
async def list_materials(course_id: str, division: str, faculty_email: str = Depends(verify_token)):
    try:
        data = rag.get_session_materials(course_id, division)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear-all")
async def clear_all(faculty_email: str = Depends(verify_token)):
    try:
        rag.clear_all_materials()
        return {"success": True, "message": "All materials cleared."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/remove-material")
async def remove_material(course_id: str, division: str, source: str, faculty_email: str = Depends(verify_token)):
    try:
        removed_chunks = rag.remove_source(course_id, division, source)
        return {"success": True, "message": f"Removed {removed_chunks} chunks for source: {source}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message": "Welcome to Classroom Companion API"}
