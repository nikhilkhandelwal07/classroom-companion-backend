# eval_data.py

RAG_CHAT_TESTS = [
    {
        "id": "chat-1",
        "course_id": "INF530",
        "division": "A",
        "question": "What is the role of product manager?"
    },
    {
        "id": "chat-2",
        "course_id": "INF530",
        "division": "A",
        "question": "What is feasibility in the context of DFV?"
    },
    {
        "id": "chat-3",
        "course_id": "INF530",
        "division": "A",
        "question": "What is the importance of user research in building a product roadmap?"
    },
    {
        "id": "chat-4",
        "course_id": "INF530",
        "division": "A",
        "question": "Can you explain the difference between B2B and B2C?"
    },
    {
        "id": "chat-5",
        "course_id": "INF530",
        "division": "A",
        "question": "Explain about the hooked model"
    }
]

FEEDBACK_SUMMARIZER_TESTS = [
    {
        "id": "feedback-1",
        "course_id": "INF501",
        "division": "A",
        "description": "Division A feedback for Business in Digital Age"
    },
    {
        "id": "feedback-2",
        "course_id": "INF530",
        "division": "A",
        "description": "Division A feedback for Maker's Lab"
    },
    {
        "id": "feedback-3",
        "course_id": "INF550",
        "division": "A",
        "description": "Division A feedback for DPM"
    }
]

DISCUSSION_FORUM_TESTS = [
    {
        "id": "forum-1",
        "course_id": "INF530",
        "division": "A",
        "question": "What is desirability in the DFV framework?",
        "has_material": True
    },
    {
        "id": "forum-2",
        "course_id": "INF530",
        "division": "A",
        "question": "Explain the feasibility aspect of DFV.",
        "has_material": True
    },
    {
        "id": "forum-4",
        "course_id": "INF530",
        "division": "A",
        "question": "what is JTBD used for?",
        "has_material": True
    },
    {
        "id": "forum-3",
        "course_id": "INF550",
        "division": "A",
        "question": "How do you calculate Customer Lifetime Value (CLV)?",
        "has_material": False
    }   
]

SESSION_PLAN_TESTS = [
    {
        "id": "plan-1",
        "course_id": "INF530",
        "division": "A",
        "session_duration": 90,
        "description": "Introduction to Product Management"
    },
    {
        "id": "plan-2",
        "course_id": "INF530",
        "division": "A",
        "session_duration": 90,
        "description": "Agile Methodologies"
    },
    {
        "id": "plan-3",
        "course_id": "INF530",
        "division": "A",
        "session_duration": 90,
        "description": "Hooked and JTBD"
    }
]
