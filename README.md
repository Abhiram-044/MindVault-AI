# MindVault AI

### Conversational AI Knowledge System (Production RAG SaaS)

> **Turn documents into an intelligent conversational memory.**

MindVault AI is a **production-ready Retrieval-Augmented Generation (RAG) platform** that enables users to upload documents and interact with them through a **real-time ChatGPT-like interface** powered by semantic search and conversational memory.

Instead of static storage, MindVault transforms documents into a **living AI knowledge base** capable of contextual reasoning, multi-turn conversations, and streaming responses.

---
# Core Features

### AI Capabilities

* Conversational RAG pipeline
* Context-aware responses
* Multi-turn chat memory
* Semantic document retrieval
* Streaming LLM responses
* Knowledge-grounded answers

---

### Document Intelligence

* Upload PDFs & documents
* Automatic chunking pipeline
* Embedding generation
* Vector similarity search
* Background processing workflow

---

### Chat Experience

* ChatGPT-style UI
* Real-time token streaming
* Session switching
* Persistent chat history
* Optimistic UI updates
* Markdown rendering

---

### SaaS Infrastructure

* JWT Authentication
* Persistent login sessions
* Auto token expiry handling
* Protected routes
* Multi-user isolation

---

### Production Engineering

* Async FastAPI backend
* SSE streaming architecture
* Zustand persistent state
* Dockerized deployment
* Cloud object storage
* Scalable RAG design

---

# System Architecture

```
                ┌──────────────────┐
                │   React Frontend  │
                │  (Chat UI SaaS)   │
                └─────────┬────────┘
                          │
                          ▼
                ┌──────────────────┐
                │   FastAPI API     │
                │ Auth + Chat       │
                └─────────┬────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                                   ▼
 Supabase Storage                    MongoDB
(Document Files)             (Chats & Sessions)
        │
        ▼
┌──────────────────────┐
│  RAG Processing Layer │
│  Chunk → Embed → Store│
└─────────┬────────────┘
          ▼
   Vector Retrieval
          ▼
     Groq LLM
          ▼
 Streaming Response (SSE)
```

---

# RAG Pipeline Flow

```
User Question
      ↓
Conversation Context
      ↓
Vector Similarity Search
      ↓
Relevant Document Chunks
      ↓
Prompt Augmentation
      ↓
LLM Generation
      ↓
Streaming Answer
```

---

# Tech Stack

## Frontend

* React + Vite + TypeScript
* Tailwind CSS
* shadcn/ui
* Zustand State Management
* Axios API Layer
* Streaming Chat Rendering

---

## Backend

* 🐍 FastAPI (Async)
* MongoDB
* Supabase Storage
* JWT Authentication
* Background Workers
* Server Sent Events (SSE)

---

## AI Stack

* Retrieval Augmented Generation (RAG)
* Semantic Embeddings
* Vector Search
* Conversational Memory
* Groq LLM Inference

---

# Project Structure

```
mindvault-ai/
│
├── backend/
│   ├─ api/
│   ├── core/
│   ├── database/
│   ├── dependencies/
│   ├── models/
│   ├── routes/
│   ├── schemas/
│   ├── services/
│   ├── storage/
│   ├── vector_store/
│   ├── main.py
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── api/
│   │   ├── store/
│   │   ├── pages/
│   │   ├── components/
│   │   └── routes/
│   └── vite.config.ts
│
└── README.md
```

---

# Local Setup

---

## 1️ Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Environment Variables

```
MONGO_URL=
JWT_SECRET=
JWT_ALGORITHM=
ACCESS_TOKEN_EXPIRE=
SUPABASE_URL=
SUPABASE_SERVICE_KEY=
SUPABASE_BUCKET=
GROQ_API_KEY=
LLM_MODEL=llama3-8b-8192
```

Run:

```bash
uvicorn main:app --reload
```

---

## 2️ Frontend

```bash
cd frontend
npm install
npm run dev
```

---

# Engineering Decisions

### Why Streaming (SSE)?

Improves perceived latency and mimics modern AI assistants like ChatGPT.

---

### Why RAG Instead of Fine-Tuning?

* Cheaper
* Updatable knowledge
* No retraining required
* Source grounded answers

---

### Why Zustand?

Minimal boilerplate with persistent SaaS-scale state control.

---

# Scalability Design

MindVault AI is designed for:

* Multi-user SaaS
* Horizontal backend scaling
* Async document processing
* Vector DB migration ready
* LLM provider abstraction

---

# Future Roadmap

* Agent workflows
* Team knowledge spaces
* Usage billing
* Vector DB scaling
* AI summarization
* Multi-document reasoning
* Enterprise RBAC

---

# Author

**Abhiram Chinta**

AI Engineer • Full Stack Developer

---

# Why This Project Matters

MindVault AI demonstrates:

* End-to-End AI System Design
* Production RAG Implementation
* Real SaaS Architecture
* Streaming AI Interfaces
* Modern Full-Stack Engineering

---

## If you like this project, star the repo!
