# OSTutorLLM

> **OSTutorLLM: A Retrieval-Augmented and Fine-Tuned Large Language Model for Personalized Operating Systems Education**

OSTutorLLM is an end-to-end educational platform designed to act as an intelligent, adaptive teaching assistant for undergraduate Operating Systems courses. It bridges the gap between high-level theoretical concepts (like process synchronization and deadlocks) and practical command-line execution.

## Key Features

* **Multi-Agent Code Sandbox:** A dual-agent framework where a Theory Agent handles conceptual queries, while a Sandbox Agent securely executes and debugs student shell scripts in an isolated container.
* **Hybrid RAG Knowledge Base:** Combines dense semantic search with sparse lexical search (BM25) across authoritative textbooks (Silberschatz), institutional lab manuals, and Linux manual pages.
* **Pedagogical Fine-Tuning:** The base LLM is instruction-tuned (QLoRA) to utilize Socratic questioning, real-world analogies, and step-by-step mathematical tracing rather than generic answer dumps.
* **Decision-Tree Prioritization:** Dynamically tracks student error rates across OS topics and automatically customizes lab scenarios and multiple-choice questions to target weak points.

## Architecture

1. **Frontend:** Streamlit / Gradio Web UI
2. **Orchestrator Layer:** LangChain / LlamaIndex Agent Router
3. **Retrieval Pipeline:** ChromaDB / FAISS with `BAAI/bge-large-en` embeddings
4. **Execution Layer:** Google Antigravity / Docker secure execution sandbox
5. **Inference:** LoRA fine-tuned Open-Source Base Model (Llama 3.3 / Qwen 3)

## Project Timeline
* **Development Phase:** September 2026 – October 2026
* **Target Completion:** End of October 2026

## Authors & Contributors

* **[Ashwin]** - [@ashwindascse-Isec](https://github.com/ashwindascse-Isec)
* **[Rohan]** - [@rohan-shah302](https://github.com/rohan-shah302)

---
*Developed as an academic research project exploring the efficacy of specialized LLMs in computer science education.*

***
