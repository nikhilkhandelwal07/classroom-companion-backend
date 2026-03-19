# evaluate.py
import requests
import json
import time
import os
import re
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from eval_data import RAG_CHAT_TESTS, FEEDBACK_SUMMARIZER_TESTS, DISCUSSION_FORUM_TESTS, SESSION_PLAN_TESTS

# Load .env
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
API_BASE_URL = "http://localhost:8001"

# Judge: Gemini Flash (via Direct Google API)
# Smarter and more cost-efficient than 8B/70B for judging
JUDGE_MODEL = "gemini-flash-latest"
JUDGE_MODEL_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{JUDGE_MODEL}:generateContent?key={GEMINI_API_KEY}"

headers = {"Content-Type": "application/json"}

def login():
    print("Logging into Classroom Companion...")
    payload = {
        "email": "dhruven@spjimr.org",
        "password": "123456789"
    }
    try:
        res = requests.post(f"{API_BASE_URL}/login", json=payload)
        data = res.json()
        if data.get("success"):
            print("Login Successful.")
            return data.get("token")
    except Exception as e:
        print(f"Login failed: {e}")
    return None

def parse_json_response(content):
    """Robustly extracts JSON from AI response patterns."""
    content = re.sub(r'```json\s*', '', content)
    content = re.sub(r'```\s*', '', content)
    content = content.strip()
    start = content.find('{')
    end = content.rfind('}') + 1
    if start == -1 or end == 0:
        return None
    return json.loads(content[start:end])

def call_judge(prompt):
    """Calls the Gemini 1.5 Flash judge model for evaluation."""
    if not GEMINI_API_KEY or "YOUR_GEMINI_API_KEY_HERE" in GEMINI_API_KEY:
        print("    [Judge] ERROR: GEMINI_API_KEY not set in .env")
        return None

    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 2048,
        }
    }
    
    for i in range(3):
        try:
            print(f"    [Judge] Calling Gemini (Attempt {i+1})...")
            res = requests.post(JUDGE_MODEL_URL, headers={"Content-Type": "application/json"}, json=payload, timeout=90)
            if res.status_code == 200:
                data = res.json()
                content = data['candidates'][0]['content']['parts'][0]['text']
                parsed = parse_json_response(content)
                if parsed:
                    return parsed
                print(f"    [Judge] Error: Failed to parse JSON. Raw summary: {content[:100]}...")
            else:
                print(f"    [Judge] Error: Status {res.status_code}. Response: {res.text[:200]}")
                time.sleep(3)
        except Exception as e:
            print(f"    [Judge] Exception: {e}")
            time.sleep(5)
    return None

def print_bar(score):
    score_val = round(score, 1)
    filled_count = int(round(score_val))
    filled = "█" * filled_count
    empty = "░" * (5 - filled_count)
    return f"{filled}{empty} {score_val}/5"

def run_evaluation():
    token = login()
    if not token:
        print("Aborting evaluation due to login failure.")
        return

    auth_headers = {"Authorization": f"Bearer {token}"}
    results = {
        "metadata": {
            "model_evaluated": "Llama 3.1 8B",
        "judge_model": "Gemini 1.5 Flash",
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "test_counts": {
                "RAG Chat": len(RAG_CHAT_TESTS),
                "Feedback Summarizer": len(FEEDBACK_SUMMARIZER_TESTS),
                "Discussion Forum": len(DISCUSSION_FORUM_TESTS),
                "Session Plan": len(SESSION_PLAN_TESTS)
            }
        },
        "components": {}
    }

    # --- COMPONENT 1: RAG Chat ---
    print("\n[1/4] Evaluating RAG Chatbot...")
    comp_results = []
    for test in RAG_CHAT_TESTS:
        print(f"  Testing {test['id']}: {test['question'][:50]}...")
        try:
            payload = {
                "question": test["question"],
                "course_id": test["course_id"],
                "division": test["division"],
                "history": []
            }
            resp = requests.post(f"{API_BASE_URL}/chat", json=payload, headers=auth_headers)
            print(f"    [Backend] /chat status: {resp.status_code}")
            
            if resp.status_code != 200:
                print(f"    [Backend] Error: {resp.text[:300]}")
                continue
                
            res = resp.json()
            answer = res.get("answer", "")
            context = res.get("context_used", "")

            prompt = f"""You are an expert AI evaluator. Score this RAG chatbot response.
Question: {test['question']}
Context: {context}
Answer: {answer}

Score each metric from 1 to 5:
FAITHFULNESS: Are all claims supported by context? (5=fully grounded, 3=partially, 1=hallucinated)
ANSWER_RELEVANCE: Does it address the question? (5=perfectly on topic, 1=irrelevant)
CONTEXT_QUALITY: Was right context retrieved? (5=excellent, 3=partial, 1=wrong chunks)
COMPLETENESS: Anything important missing? (5=comprehensive, 1=missing key info)

Return ONLY JSON:
{{
  "faithfulness": 0, "answer_relevance": 0, "context_quality": 0, "completeness": 0,
  "faithfulness_reason": "...", "answer_relevance_reason": "...", "context_quality_reason": "...", "completeness_reason": "..."
}}"""
            eval_res = call_judge(prompt)
            if eval_res:
                eval_res["overall"] = round(sum([eval_res[k] for k in ["faithfulness", "answer_relevance", "context_quality", "completeness"]]) / 4, 2)
                eval_res["id"] = test["id"]
                comp_results.append(eval_res)
                print(f"    [Result] Overall: {eval_res['overall']}/5")
            else:
                print(f"    [Judge] Failed to evaluate test {test['id']}")
            time.sleep(3)
        except Exception as e:
            print(f"    ERROR in RAG Chat loop: {e}")

    results["components"]["RAG Chat"] = comp_results

    # --- COMPONENT 2: Feedback Summarizer ---
    print("\n[2/4] Evaluating Feedback Summarizer...")
    comp_results = []
    for test in FEEDBACK_SUMMARIZER_TESTS:
        print(f"  Testing {test['id']}: {test['description']}...")
        try:
            params = {"course_id": test["course_id"], "division": test["division"]}
            resp = requests.get(f"{API_BASE_URL}/feedback", params=params, headers=auth_headers)
            print(f"    [Backend] /feedback status: {resp.status_code}")
            
            if resp.status_code != 200:
                print(f"    [Backend] Error: {resp.text[:300]}")
                continue
                
            res = resp.json()
            # Handle actual response structure
            feedback_data = res.get("ai_analysis", {})
            summary = feedback_data.get("summary", "")
            working_well = feedback_data.get("working_well", [])
            areas = feedback_data.get("areas_to_improve", [])
            comments = [c["text"] for c in res.get("comments", [])]
            ratings = [c["rating"] for c in res.get("comments", [])]

            if not comments:
                print(f"    [Backend] SKIP: No feedback data found.")
                continue

            prompt = f"""Score this Feedback Summarizer response.
Input Feedback: {[f'Rating {r}/5: {c}' for r, c in zip(ratings, comments)]}
AI Summary: {summary}
Working Well: {working_well}
Areas to Improve: {areas}

Score each metric from 1 to 5:
ACCURACY: Does summary reflect actual feedback? (5=perfectly accurate, 3=mostly, 1=misleading)
SPECIFICITY: Are points specific to comments or generic? (5=highly specific, 1=completely generic)
ACTIONABILITY: Can faculty act on improvements? (5=concrete steps, 1=vague)
BALANCE: Positive vs Negative fair representation? (5=well balanced, 1=one-sided)

Return ONLY JSON:
{{
  "accuracy": 0, "specificity": 0, "actionability": 0, "balance": 0,
  "accuracy_reason": "...", "specificity_reason": "...", "actionability_reason": "...", "balance_reason": "..."
}}"""
            eval_res = call_judge(prompt)
            if eval_res:
                eval_res["overall"] = round(sum([eval_res[k] for k in ["accuracy", "specificity", "actionability", "balance"]]) / 4, 2)
                eval_res["id"] = test["id"]
                comp_results.append(eval_res)
                print(f"    [Result] Overall: {eval_res['overall']}/5")
            else:
                print(f"    [Judge] Failed to evaluate test {test['id']}")
            time.sleep(3)
        except Exception as e:
            print(f"    ERROR in Feedback loop: {e}")

    results["components"]["Feedback Summarizer"] = comp_results

    # --- COMPONENT 3: Discussion Forum ---
    print("\n[3/4] Evaluating Discussion Forum AI...")
    comp_results = []
    for test in DISCUSSION_FORUM_TESTS:
        print(f"  Testing {test['id']}: {test['question'][:50]}...")
        try:
            payload = {
                "question_id": "eval-test",
                "question_text": test["question"],
                "course_id": test["course_id"],
                "division": test["division"]
            }
            resp = requests.post(f"{API_BASE_URL}/suggest-answer", json=payload, headers=auth_headers)
            print(f"    [Backend] /suggest-answer status: {resp.status_code}")
            
            if resp.status_code != 200:
                print(f"    [Backend] Error: {resp.text[:300]}")
                continue
                
            res = resp.json()
            answer = res.get("ai_suggestion", "")
            context = res.get("context_used", "")
            used_mat = res.get("used_material", False)

            prompt = f"""Score this Discussion Forum AI Suggestion.
Question: {test['question']}
Context: {context if context else "None uploaded"}
AI Suggestion: {answer}
System Flag (Used Material): {used_mat}

Score each metric from 1 to 5:
HELPFULNESS: Usefulness for student explanation? (5=very helpful, 1=not helpful)
GROUNDEDNESS: Based on material if available, or correctly flagged? (5=perfect, 1=hallucinated)
CLARITY: Clear and easy to understand? (5=very clear, 1=confusing)
APPROPRIATENESS: Right tone for business school? (5=perfectly appropriate, 1=unprofessional)

Return ONLY JSON:
{{
  "helpfulness": 0, "groundedness": 0, "clarity": 0, "appropriateness": 0,
  "helpfulness_reason": "...", "groundedness_reason": "...", "clarity_reason": "...", "appropriateness_reason": "..."
}}"""
            eval_res = call_judge(prompt)
            if eval_res:
                eval_res["overall"] = round(sum([eval_res[k] for k in ["helpfulness", "groundedness", "clarity", "appropriateness"]]) / 4, 2)
                eval_res["id"] = test["id"]
                comp_results.append(eval_res)
                print(f"    [Result] Overall: {eval_res['overall']}/5")
            else:
                print(f"    [Judge] Failed to evaluate test {test['id']}")
            time.sleep(3)
        except Exception as e:
            print(f"    ERROR in Discussion loop: {e}")

    results["components"]["Discussion Forum"] = comp_results

    # --- COMPONENT 4: Session Plan ---
    print("\n[4/4] Evaluating Session Plan Generator...")
    comp_results = []
    for test in SESSION_PLAN_TESTS:
        print(f"  Testing {test['id']}: {test['description']}...")
        try:
            payload = {
                "course_id": test["course_id"],
                "division": test["division"],
                "session_duration": test["session_duration"]
            }
            resp = requests.post(f"{API_BASE_URL}/generate-session-plan", json=payload, headers=auth_headers)
            print(f"    [Backend] /generate-session-plan status: {resp.status_code}")
            
            if resp.status_code != 200:
                print(f"    [Backend] Error: {resp.text[:300]}")
                continue
                
            res = resp.json()
            blocks = res.get("blocks", [])

            prompt = f"""Score this AI-generated Session Plan.
Topic: {test['description']}
Duration: {test['session_duration']} min
Plan Structure: {json.dumps(blocks, indent=2)}

Score each metric from 1 to 5:
STRUCTURE: Logical time blocks? (5=perfect flow, 1=chaotic)
CONTENT_RELEVANCE: Based on course context? (5=highly relevant, 1=generic)
PRACTICALITY: Can faculty actually run this? (5=very practical, 1=unrealistic)
TIME_ALLOCATION: Realistic timing? (5=perfect, 1=mismatched)

Return ONLY JSON:
{{
  "structure": 0, "content_relevance": 0, "practicality": 0, "time_allocation": 0,
  "structure_reason": "...", "content_relevance_reason": "...", "practicality_reason": "...", "time_allocation_reason": "..."
}}"""
            eval_res = call_judge(prompt)
            if eval_res:
                eval_res["overall"] = round(sum([eval_res[k] for k in ["structure", "content_relevance", "practicality", "time_allocation"]]) / 4, 2)
                eval_res["id"] = test["id"]
                comp_results.append(eval_res)
                print(f"    [Result] Overall: {eval_res['overall']}/5")
            else:
                print(f"    [Judge] Failed to evaluate test {test['id']}")
            time.sleep(3)
        except Exception as e:
            print(f"    ERROR in Session Plan loop: {e}")

    results["components"]["Session Plan"] = comp_results

    # Save Results
    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # FINAL REPORT
    print_report(results)

def print_report(results):
    print("\n" + "="*60)
    print("      CLASSROOM COMPANION - AI EVALUATION REPORT")
    print("="*60)
    print(f"Date: {results['metadata']['date']}")
    print(f"Model Evaluated: {results['metadata']['model_evaluated']}")
    print(f"Judge Model: {results['metadata']['judge_model']}")
    print("-" * 60)

    summary_table = []

    components_metrics = {
        "RAG Chat": ["faithfulness", "answer_relevance", "context_quality", "completeness"],
        "Feedback Summarizer": ["accuracy", "specificity", "actionability", "balance"],
        "Discussion Forum": ["helpfulness", "groundedness", "clarity", "appropriateness"],
        "Session Plan": ["structure", "content_relevance", "practicality", "time_allocation"]
    }

    for comp_name, metrics in components_metrics.items():
        data = results["components"].get(comp_name, [])
        if not data: continue
        
        print(f"\n{comp_name.upper()}")
        print("-" * len(comp_name))
        
        comp_scores = []
        for metric in metrics:
            scores = [r[metric] for r in data if metric in r]
            if scores:
                avg_m = sum(scores) / len(scores)
                print(f"  {metric.replace('_', ' ').title():<20} {print_bar(avg_m)}")
        
        overall_avg = sum(r["overall"] for r in data) / len(data)
        print(f"  {'OVERALL SCORE':<20} {print_bar(overall_avg)}")
        summary_table.append((comp_name, overall_avg))

    print("\n" + "="*60)
    print(f"{'COMPONENT':<30} {'SCORE':<20}")
    print("-" * 60)
    grand_total = 0
    for comp, score in summary_table:
        print(f"{comp:<30} {print_bar(score)}")
        grand_total += score
    
    if summary_table:
        grand_avg = grand_total / len(summary_table)
        print("-" * 60)
        print(f"{'GRAND AVERAGE':<30} {print_bar(grand_avg)}")
    print("="*60)
    print("Full audit logs saved to eval_results.json")

if __name__ == "__main__":
    run_evaluation()
