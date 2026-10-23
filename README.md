# OSTutorLLM: A Retrieval-Grounded Learning Assistant for Operating Systems

**Vellore Institute of Technology**  
**Course Code:** BCSE303P — Operating Systems Lab  
**Authors:** Ashwin Das (24BCI0254) & Rohan Shah (24BCE2121)  
**Academic Year:** 2026-2027  

---

## 📖 Overview

**OSTutorLLM** is an integrated, subject-focused study environment designed for students and educators of Operating Systems. Instead of treating every task as open-ended chat, OSTutorLLM cleanly separates:
1. **Generative Language Tasks**: Handled by LLMs (Groq / Gemini) grounded strictly in retrieved course textbook passages via hybrid Okapi BM25 + TF-IDF ranking with page-level provenance.
2. **Deterministic Algorithm Tasks**: Handled by rigorous, pure Python simulation engines that guarantee repeatable, testable mathematical outcomes for CPU scheduling, page replacement, and deadlock avoidance without model hallucinations.

---

## 🏗️ System Architecture & Code Structure

The project adheres to a strict separation of concerns across modules in `src/`:

```
OStutLLM/
├── app.py                      # Main Streamlit application UI & navigation
├── .env                        # API keys (Groq, Gemini)
├── requirements.txt            # Python dependencies
├── data/
│   ├── library/                # Ingested document passage stores
│   ├── library_manifest.json   # Document library metadata and hashes
│   ├── progress_records.json   # Spaced repetition & quiz history
│   └── default_materials/      # Curated standard Operating Systems chapters
├── src/
│   ├── config.py               # Multi-provider config & .env loader
│   ├── ingestion.py            # PDF/TXT extraction, paragraph chunking, page metadata
│   ├── retrieval.py            # Hybrid Okapi BM25 + TF-IDF search & deduplication
│   ├── library.py              # Library persistence, manifest, deduplication
│   ├── tutor.py                # Grounded Tutor RAG, prompt engine, provider calls
│   ├── study_tools.py          # Active recall flashcards & multiple-choice quizzes
│   ├── calculators.py          # Deterministic CPU, Page Replacement & Banker's logic
│   ├── os_lab.py               # Interactive OS Lab simulations & Gantt charts
│   ├── progress.py             # Progress tracking & explainable revision queue
│   ├── citation_validator.py   # Grounding score and source reference verifier
│   ├── source_analysis.py      # Mind map SVG generator & syllabus coverage evaluator
│   ├── evaluation.py           # Benchmark suite runner on standard OS questions
│   └── evaluation_dashboard.py # Streamlit UI for evaluation metrics & latency
└── tests/
    ├── test_calculators.py     # Deterministic algorithm unit tests
    ├── test_retrieval.py       # Hybrid retrieval & deduplication tests
    └── test_study_tools.py     # Study tools, SVG rendering, syllabus tests
```

---

## 🚀 Key Features

### 1. 🧑‍🏫 Grounded Conversational Tutor
- Answers are synthesized strictly from retrieved textbook passages.
- Includes exact page-level citations (`[Document, Page X]`) with quote snippets.
- Tunable answer lengths: **Short** (definitions), **Medium** (concept + example), **Long** (detailed walkthrough).
- Bounded conversation history for follow-ups ("What is its main advantage?").
- Transparently admits when selected documents have insufficient evidence.

### 2. Integrated Study Tools
- **Flashcards:** Generates distinct front-and-back pairs with stable IDs. Features an interactive flip card UI, answer reveal, and "Mark as Mastered" / "Mark as Missed" buttons that feed spaced repetition.
- **Multiple-Choice Quiz:** 4 unique options, verified single correct answer, distractor validation, explanations, and immediate score feedback.
- **Concept Mind Map:** Generates clean SVG diagrams distinguishing **source-supported** relationships (solid cyan lines with citations) from **inferred/organizational** groupings (dashed violet lines).
- **Syllabus Coverage:** Evaluates course textbooks against syllabus topics (or custom entered topics) while strictly excluding the syllabus itself from evidence to prevent circular reasoning. Categorizes topics into *Covered*, *Partially Covered*, and *Not Covered*.

### 3. Deterministic OS Laboratory
- **CPU Scheduling:** FCFS, SJF (Non-preemptive), SRTF (Preemptive SJF), Priority (Non-preemptive & Preemptive), Round Robin (with configurable time quantum). Produces an interactive Gantt chart, step-by-step transition trace, and calculated metrics (Completion Time, Turnaround Time, Waiting Time, Response Time, CPU Utilization, Throughput).
- **Page Replacement:** FIFO, LRU, and Belady's Optimal algorithms with step-by-step memory frame grids. Includes a side-by-side comparison mode running all three algorithms on the same reference string.
- **Deadlock Avoidance (Banker's Algorithm):** Derives the Need Matrix (`Need = Max - Allocation`), performs Dijkstra's safety algorithm to discover safe execution sequences (`<P1 -> P3 -> P4 -> P0 -> P2>`), and provides a Resource-Request simulator for incoming process claims.

### 4. Progress and Revision
- Tracks flashcard reviews and quiz performance per topic.
- Identifies weak topics and generates an explainable revision queue.

---


## 🛠️ Installation & Setup

1. **Activate virtual environment:**
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables (`.env`):**
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

4. **Run the Streamlit Application:**
   ```bash
   python -m streamlit run app.py
   ```
   Open your browser at `http://localhost:8501`.

5. **Run the Test Suite:**
   ```bash
   python -m pytest tests/
   ```
