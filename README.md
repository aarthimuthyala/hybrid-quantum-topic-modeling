# Hybrid Quantum–Classical Optimization for Topic Modeling & Document Clustering

**Blueprint v1.0 · Final Year Research Project · 2025–2026**

> A research system combining QAOA quantum optimization with classical NLP (LDA/NMF) for document topic modeling and clustering, with noise-aware simulation and Zero-Noise Extrapolation (ZNE).

---

## Table of Contents

1. [Project Architecture](#1-project-architecture)
2. [Prerequisites](#2-prerequisites)
3. [Repository Structure](#3-repository-structure)
4. [Python Environment Setup](#4-python-environment-setup)
5. [Backend Setup & Startup](#5-backend-setup--startup)
6. [Frontend Setup & Startup](#6-frontend-setup--startup)
7. [Full Execution Workflow](#7-full-execution-workflow)
8. [API Reference](#8-api-reference)
9. [Deployment Guide](#9-deployment-guide)
10. [Troubleshooting](#10-troubleshooting)
11. [Team Assignments](#11-team-assignments)

---

## 1. Project Architecture

```
User Browser (React, port 3000)
        │
        │  REST API (HTTP/JSON)
        ▼
FastAPI Backend (port 8000)
        │
   ┌────┴─────────────────────────┐
   │                              │
NLP Pipeline              Quantum Engine
(LDA · NMF · SBERT)      (QAOA · QUBO · AER)
   │                              │
   └────────────┬─────────────────┘
                │
        Hybrid Orchestrator
        (Warm-start + COBYLA loop)
                │
         SQLite Database
         + File Storage
```

**Layers (Blueprint §1):**

| Layer | Role | Tech |
|-------|------|------|
| L1 | Data Ingestion & Preprocessing | spaCy · NLTK · HuggingFace |
| L2 | Classical NLP Baseline | scikit-learn · Gensim |
| L3 | Quantum–Classical Hybrid Engine | Qiskit 1.x · Qiskit Aer |
| L4 | Evaluation, Visualisation & API | FastAPI · React · Recharts |

---

## 2. Prerequisites

### System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| OS | Windows 10 / Ubuntu 20.04 / macOS 12 | Ubuntu 22.04 / macOS 14 |
| Python | 3.11 | 3.11.x |
| Node.js | 18.x | 20.x LTS |
| npm | 9.x | 10.x |
| RAM | 8 GB | 16 GB |
| Disk | 5 GB free | 10 GB free |

### Check Versions

**Windows (Command Prompt / PowerShell):**
```bat
python --version
node --version
npm --version
```

**Linux / macOS:**
```bash
python3 --version
node --version
npm --version
```

> **Python note:** Qiskit 1.x and qiskit-aer require Python 3.8–3.11. Python 3.12 is not yet fully supported by all Qiskit packages. Use Python 3.11 to avoid compatibility issues.

---

## 3. Repository Structure

```
hybrid-quantum-topic-modeling/
├── backend/
│   ├── main.py                  # FastAPI entry point
│   ├── config.py                # Settings (reads .env)
│   ├── requirements.txt         # Backend-only dependencies
│   ├── .env.example             # Environment variable template
│   ├── api/                     # Route handlers
│   ├── nlp/                     # Preprocessing, LDA, NMF, embeddings
│   ├── classical/               # K-Means, evaluator
│   ├── quantum/                 # QUBO encoder, QAOA circuit, optimizer
│   ├── noise/                   # Noise models, AER simulator, ZNE
│   ├── hybrid/                  # Orchestrator, warm-start, cost function
│   ├── db/                      # SQLAlchemy models, CRUD
│   ├── utils/                   # Logger, metrics, visualiser, exporter
│   └── data/                    # Datasets, uploads, results (git-ignored)
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # Router shell + global design system
│   │   ├── main.jsx             # React DOM mount
│   │   ├── services/api.js      # Axios API service (all §4 endpoints)
│   │   ├── components/
│   │   │   ├── Layout/Navbar.jsx
│   │   │   └── Results/
│   │   │       ├── MetricPanel.jsx
│   │   │       └── TopicViewer.jsx
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx
│   │   │   ├── Pipeline.jsx
│   │   │   ├── Results.jsx
│   │   │   └── QuantumLab.jsx
│   │   └── styles/main.css
│   ├── package.json
│   ├── vite.config.js
│   └── README.md
├── notebooks/                   # Jupyter research notebooks
├── tests/                       # Unit + integration tests
├── requirements.txt             # Root requirements (full project)
└── README.md                    # This file
```

---

## 4. Python Environment Setup

### Step 4.1 — Create Virtual Environment

**Windows:**
```bat
cd hybrid-quantum-topic-modeling
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS:**
```bash
cd hybrid-quantum-topic-modeling
python3.11 -m venv .venv
source .venv/bin/activate
```

You should see `(.venv)` in your terminal prompt after activation.

### Step 4.2 — Upgrade pip

```bash
pip install --upgrade pip setuptools wheel
```

### Step 4.3 — Install Python Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** Qiskit Aer compilation can take 2–5 minutes on first install. This is normal.

### Step 4.4 — Download NLP Models

Run these once after pip install:

```bash
# NLTK data (punkt tokenizer, stopwords, wordnet)
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# spaCy English model
python -m spacy download en_core_web_sm
```

### Step 4.5 — Verify Quantum Install

```bash
python -c "import qiskit; from qiskit_aer import AerSimulator; print('Qiskit OK:', qiskit.__version__)"
```

Expected output: `Qiskit OK: 1.1.0`

---

## 5. Backend Setup & Startup

### Step 5.1 — Configure Environment

```bash
cd backend
cp .env.example .env
```

Open `.env` and review defaults. Key settings:

```env
HOST=0.0.0.0
PORT=8000
DEBUG=True
DATABASE_URL=sqlite:///./data/quantum_topic.db
QUANTUM_BACKEND=aer_simulator
ENABLE_NOISE=True
DATASET_NAME=20newsgroups
DATASET_SUBSET_SIZE=200
```

### Step 5.2 — Start the Backend Server

**From the `backend/` directory:**

**Windows:**
```bat
cd backend
python main.py
```

**Linux / macOS:**
```bash
cd backend
python main.py
```

**Or using uvicorn directly (any OS):**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Expected startup output:**
```
=== Hybrid Quantum-Classical Backend Starting ===
Database tables created / verified.
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

### Step 5.3 — Verify Backend

Open in browser: **http://localhost:8000**

Expected JSON response:
```json
{
  "project": "Hybrid Quantum-Classical Topic Modeling",
  "version": "1.0.0",
  "status": "running",
  "docs": "/docs"
}
```

Interactive API docs: **http://localhost:8000/docs**

---

## 6. Frontend Setup & Startup

### Step 6.1 — Install Node Dependencies

**Open a new terminal (keep backend running in the first).**

```bash
cd frontend
npm install
```

> First install downloads ~150 MB of node_modules. Subsequent installs use cache.

### Step 6.2 — Configure Environment

```bash
cp .env.example .env
```

Default `.env` content:
```env
VITE_API_BASE_URL=http://localhost:8000/api/v1
```

### Step 6.3 — Start the Development Server

```bash
npm run dev
```

**Expected output:**
```
  VITE v5.x.x  ready in 300ms

  ➜  Local:   http://localhost:3000/
  ➜  Network: http://0.0.0.0:3000/
```

Open **http://localhost:3000** in your browser.

> The Vite dev server proxies all `/api/*` requests to `http://localhost:8000` automatically (configured in `vite.config.js`). No CORS issues during development.

### Step 6.4 — Build for Production

```bash
npm run build
```

Output is written to `frontend/dist/`. Serve with:

```bash
npm run preview
```

---

## 7. Full Execution Workflow

This is the recommended startup sequence for the complete system:

### Terminal 1 — Backend

```bash
# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Start backend
cd backend
python main.py
```

### Terminal 2 — Frontend

```bash
cd frontend
npm run dev
```

### Browser — Research Console

1. Open **http://localhost:3000**
2. The Navbar shows **API ONLINE** (green) when backend is connected
3. Use **Quick Launch** on Dashboard to load a dataset and run the pipeline
4. Monitor progress on the **Pipeline** page
5. View metrics and export results on the **Results** page
6. Explore noise models and QAOA theory on the **Quantum Lab** page

### Pipeline Execution Sequence (Blueprint §5.1)

```
Stage 01  Raw Corpus Ingest        →  POST /api/v1/documents/load-dataset
Stage 02  Text Preprocessing       →  POST /api/v1/ingest/preprocess
Stage 03  Tokenization & Vocab     →  (internal to backend)
Stage 04  Classical Baseline       →  POST /api/v1/classical/lda/train
Stage 05  Cost Hamiltonian Build   →  (internal — hybrid/cost_function.py)
Stage 06  Quantum Optimization     →  POST /api/v1/quantum/qaoa/run
Stage 07  Hybrid Clustering        →  POST /api/v1/hybrid/run
Stage 08  Evaluation & Report      →  GET  /api/v1/eval/report/{run_id}
```

---

## 8. API Reference

**Base URL:** `http://localhost:8000/api/v1`

Full interactive docs at: **http://localhost:8000/docs** (Swagger UI)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | System health check |
| POST | `/documents/load-dataset` | Load a Blueprint dataset |
| POST | `/documents/bulk` | Upload custom documents |
| POST | `/ingest/preprocess` | Preprocess a corpus |
| POST | `/classical/lda/train` | Train LDA model |
| POST | `/classical/nmf/train` | Train NMF model |
| POST | `/classical/cluster` | Run K-Means clustering |
| POST | `/quantum/qaoa/run` | Execute QAOA optimisation |
| POST | `/quantum/noise/build` | Build a noise model |
| GET | `/quantum/job/{job_id}` | Poll async quantum job |
| POST | `/hybrid/run` | Launch full hybrid pipeline |
| GET | `/hybrid/run/{run_id}` | Fetch hybrid run results |
| POST | `/hybrid/compare` | Classical vs hybrid benchmark |
| GET | `/eval/coherence/{model_id}` | Topic coherence metrics |
| GET | `/eval/cluster/{cluster_id}` | Cluster quality metrics |
| GET | `/results/{job_id}/export` | Export results (JSON/CSV) |

---

## 9. Deployment Guide

### Option A — Local Development (Default)

Follow Sections 5 and 6 above. Both services run locally with hot-reload enabled.

---

### Option B — Docker Compose

**Prerequisites:** Docker Desktop installed and running.

**Step 1 — Build and start all services:**

```bash
docker-compose up --build
```

**Step 2 — Access services:**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

**Step 3 — Stop services:**

```bash
docker-compose down
```

**Step 4 — Rebuild after code changes:**

```bash
docker-compose up --build --force-recreate
```

---

### Option C — Manual Production Deployment

#### Backend (Production)

```bash
# Install with production dependencies only
pip install -r requirements.txt

# Set production environment
export DEBUG=False
export HOST=0.0.0.0
export PORT=8000

# Run with multiple workers (Linux only)
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2

# Windows — single worker only
uvicorn main:app --host 0.0.0.0 --port 8000
```

#### Frontend (Production Build)

```bash
cd frontend

# Set production API URL
echo "VITE_API_BASE_URL=https://your-api-domain.com/api/v1" > .env

# Build static files
npm run build

# Preview locally (optional)
npm run preview
```

The `frontend/dist/` folder contains the compiled static site. Deploy it to any static host (Nginx, Apache, Vercel, Netlify, GitHub Pages).

#### Nginx Config Example (Linux)

```nginx
# /etc/nginx/sites-available/hqc-frontend
server {
    listen 80;
    server_name your-domain.com;
    root /var/www/hqc/frontend/dist;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /api/ {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## 10. Troubleshooting

### Backend Issues

**`ModuleNotFoundError: No module named 'qiskit_aer'`**
```bash
pip install qiskit-aer==0.14.1
```

**`ModuleNotFoundError: No module named 'pydantic_settings'`**
```bash
pip install pydantic-settings==2.2.1
```

**`LookupError: Resource punkt not found`**
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

**`OSError: [E050] Can't find model 'en_core_web_sm'`**
```bash
python -m spacy download en_core_web_sm
```

**Backend starts but shows database errors**
```bash
# Delete the SQLite database and let it recreate
rm backend/data/quantum_topic.db   # Linux/macOS
del backend\data\quantum_topic.db  # Windows
python main.py
```

**Port 8000 already in use**

Windows:
```bat
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

Linux/macOS:
```bash
lsof -ti:8000 | xargs kill -9
```

**Qiskit Aer simulation is very slow**

This is expected for larger circuits. Reduce `QUANTUM_SHOTS` in `.env`:
```env
QUANTUM_SHOTS=256
```

Or reduce `DATASET_SUBSET_SIZE` for the quantum experiment subset:
```env
DATASET_SUBSET_SIZE=50
```

---

### Frontend Issues

**`npm install` fails with EACCES (Linux/macOS)**
```bash
sudo chown -R $(whoami) ~/.npm
npm install
```

**`npm install` fails with Python build errors (Windows)**

Install Windows Build Tools:
```bat
npm install --global windows-build-tools
```
Or install Visual Studio Build Tools from Microsoft.

**Port 3000 already in use**

Windows:
```bat
netstat -ano | findstr :3000
taskkill /PID <PID> /F
```

Linux/macOS:
```bash
lsof -ti:3000 | xargs kill -9
```

**Navbar shows "API OFFLINE" (red)**

1. Confirm the backend is running: `http://localhost:8000`
2. Check the browser console for CORS errors
3. Confirm `.env` has the correct `VITE_API_BASE_URL`
4. Restart the Vite dev server after editing `.env`

**Blank page after `npm run build`**

Ensure `base` path is set correctly for your deployment host. For a subdirectory deployment, add to `vite.config.js`:
```js
base: '/your-subdirectory/',
```

**`axios` network errors in browser**

The Vite dev proxy forwards `/api/*` to the backend. If you access the built frontend directly (not via `npm run dev`), requests go to `VITE_API_BASE_URL`. Ensure the backend URL is reachable and CORS is configured.

---

### Environment Issues

**`python` not found on Windows — use `py` instead**
```bat
py -3.11 -m venv .venv
```

**Virtual environment not activating on Windows (execution policy)**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.venv\Scripts\Activate.ps1
```

**Wrong Python version is being used**

Windows:
```bat
py -3.11 -m venv .venv
```

Linux/macOS:
```bash
python3.11 -m venv .venv
# or, if using pyenv:
pyenv local 3.11.9
python -m venv .venv
```

---

## 11. Team Assignments

| Team | Owns | Blueprint Section |
|------|------|-------------------|
| T-1 | `src/ingestion/`, `data/`, `config/base_config.yaml` | §3.1 |
| T-2 | `src/classical/`, `config/classical_config.yaml` | §3.2 |
| T-3 | `src/quantum/`, `config/quantum_config.yaml`, `noise_config.yaml` | §3.3 |
| T-4 | `src/hybrid/`, `src/evaluation/`, `src/api/`, `outputs/` | §3.4–3.5 |
| T-6 | `frontend/`, `README.md`, `requirements.txt`, docs | §4 (UI) |

**Contract Freeze Protocol (§10.3):** Schema changes require an Architecture Decision Record filed as a GitHub Issue tagged `[ADR]`, reviewed by the Chief Architect, and announced 48 hours before merging.

---

## Quick Reference

```bash
# ── Full system startup (run each in a separate terminal) ──────────────────────

# Terminal 1: Backend
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows
cd backend && python main.py

# Terminal 2: Frontend
cd frontend && npm run dev

# ── URLs ──────────────────────────────────────────────────────────────────────
# Research Console:  http://localhost:3000
# API Root:          http://localhost:8000
# Swagger UI:        http://localhost:8000/docs

# ── Tests ─────────────────────────────────────────────────────────────────────
pytest tests/ -v

# ── Production build ──────────────────────────────────────────────────────────
cd frontend && npm run build
```

---

*HQC Topic Modeling Project · Blueprint v1.0 · 2025–2026*