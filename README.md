---
title: Classroom Companion API
emoji: 🎓
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Classroom Companion API

Backend API for Classroom Companion — an AI teaching assistant for SPJIMR faculty.

## Core Features
- **Attendance Insights**: Real-time status and at-risk student detection.
- **RAG Powered Knowledge**: Upload PDFs or URLs to build a searchable knowledge base per course context.
- **AI Summary & Planning**: Automated generation of session summaries and detailed teaching plans.
- **Sentiment Analysis**: Deep-dive into student feedback using specialized NLP models.

## API Endpoints

### 🔐 Authentication
- `POST /login`: Faculty authentication and session token generation.

### 📊 Attendance & Analytics
- `GET /attendance`: Retrieve student participation data.
- `POST /email-attendance`: Notify at-risk students via automated email.
- `GET /feedback`: AI-powered sentiment analysis of student survey results.

### 📚 Course Materials (RAG)
- `POST /upload-material`: Upload PDF documents to the session.
- `POST /add-url`: Scrape and index web content for the session.
- `GET /list-materials`: Sync UI with currently indexed materials.
- `DELETE /remove-material`: Remove a specific source from the index.
- `POST /clear-material`: Purge all materials for a specific course/division.
- `POST /clear-all`: Global cleanup of all faculty materials (used on logout).

### 🤖 AI Generation
- `POST /generate-summary`: Create an AI summary from uploaded materials.
- `POST /generate-session-plan`: Generate a pedagogical session plan.
- `POST /chat`: Interactive RAG-based context-aware chatbot.
- `POST /email-material`: Distribute summaries and materials to students via email.

### 🛠️ System
- `GET /health`: System health and status check.
- `GET /`: API welcome message.

## Setup
Developed with **FastAPI**, **LangChain**, and **FAISS**. Optimized for deployment on HuggingFace Spaces.
