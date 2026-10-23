"""
OSTutorLLM - A Retrieval-Grounded Learning Assistant for Operating Systems
Course Code: BCSE303P - Operating Systems Lab | Vellore Institute of Technology
Authors: Ashwin Das (24BCI0254) & Rohan Shah (24BCE2121)
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import time
from typing import List, Dict, Any

from src.config import (
    get_groq_key,
    get_gemini_key,
    get_masked_key,
    DEFAULT_GROQ_MODEL,
    DEFAULT_GEMINI_MODEL
)
from src.library import DocumentLibrary
from src.retrieval import hybrid_search
from src.tutor import generate_tutor_response
from src.study_tools import generate_flashcards, generate_quiz
from src.source_analysis import (
    generate_mind_map,
    evaluate_syllabus_coverage,
    parse_custom_syllabus_text,
    DEFAULT_OS_SYLLABUS
)
from src.progress import ProgressTracker
from src.os_lab import render_os_lab

# Set page configuration
st.set_page_config(
    page_title="OSTutorLLM - Operating Systems Lab",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Mode, Glassmorphism, Modern Typography)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', system-ui, sans-serif;
    }
    
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
    }
    
    /* Top Header Banner */
    .header-banner {
        background: linear-gradient(135deg, rgba(79, 70, 229, 0.15) 0%, rgba(6, 182, 212, 0.15) 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(8px);
    }
    
    .app-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #818cf8 0%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.02em;
    }
    
    .app-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: 6px;
        font-weight: 500;
    }
    
    /* Card Glass Styling */
    .glass-card {
        background: rgba(17, 24, 39, 0.7);
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    
    /* Citations Block */
    .citation-badge {
        display: inline-block;
        background: rgba(14, 165, 233, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    
    .citation-accordion {
        background-color: #0f172a;
        border-left: 3px solid #0284c7;
        padding: 10px 14px;
        border-radius: 4px;
        margin-top: 8px;
        font-size: 0.85rem;
        color: #cbd5e1;
    }
    
    /* Flashcard UI */
    .flashcard-box {
        background: linear-gradient(145deg, #1e1b4b 0%, #0f172a 100%);
        border: 2px solid #6366f1;
        border-radius: 16px;
        padding: 32px;
        text-align: center;
        min-height: 220px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.3);
        margin: 16px 0;
    }
    
    .flashcard-prompt {
        font-size: 1.35rem;
        font-weight: 700;
        color: #e0e7ff;
        margin-bottom: 12px;
    }
    
    .flashcard-answer {
        font-size: 1.05rem;
        color: #38bdf8;
        background: rgba(6, 182, 212, 0.1);
        padding: 14px 20px;
        border-radius: 10px;
        border: 1px solid rgba(6, 182, 212, 0.3);
        margin-top: 14px;
        line-height: 1.5;
    }
    
    /* Metrics overrides */
    [data-testid="stMetricValue"] {
        font-weight: 700;
        color: #38bdf8;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Singletons & Session State
if "library" not in st.session_state:
    st.session_state.library = DocumentLibrary()

if "progress_tracker" not in st.session_state:
    st.session_state.progress_tracker = ProgressTracker()

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "fc_deck" not in st.session_state:
    st.session_state.fc_deck = []
if "fc_index" not in st.session_state:
    st.session_state.fc_index = 0
if "fc_show_back" not in st.session_state:
    st.session_state.fc_show_back = False

if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = []
if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False
if "quiz_selections" not in st.session_state:
    st.session_state.quiz_selections = {}

if "mind_map_res" not in st.session_state:
    st.session_state.mind_map_res = None

if "coverage_res" not in st.session_state:
    st.session_state.coverage_res = None

if "custom_syllabus_text" not in st.session_state:
    st.session_state.custom_syllabus_text = (
        "Process Scheduling: First Come First Served, Shortest Job First, Round Robin\n"
        "Deadlock Management: Coffman Conditions, Banker's Safety Algorithm, Prevention\n"
        "Memory Management: Paging Architecture, Page Replacement FIFO LRU, Demand Paging"
    )


lib: DocumentLibrary = st.session_state.library
tracker: ProgressTracker = st.session_state.progress_tracker


# ============================================================================
# SIDEBAR: Providers, Model, Document Library Manager, and Metadata
# ============================================================================

with st.sidebar:
    st.markdown("### Provider and Configuration")

    groq_k = get_groq_key()
    gemini_k = get_gemini_key()

    provider_options = []
    if groq_k:
        provider_options.append("Groq")
    if gemini_k:
        provider_options.append("Gemini")
    if not provider_options:
        provider_options = ["None (Demo Mode)"]

    selected_provider = st.selectbox(
        "Active Generation Provider",
        provider_options,
        index=0,
        help="Groq provides ultra-fast inference with Llama/Qwen models; Gemini provides Google multimodal generation."
    )

    if selected_provider == "Groq":
        st.caption(f"Groq Key: `{get_masked_key(groq_k)}` (Status: Active)")
        model_name = st.selectbox("Groq Model", [DEFAULT_GROQ_MODEL, "openai/gpt-oss-120b", "openai/gpt-oss-20b"], index=0)
    elif selected_provider == "Gemini":
        st.caption(f"Gemini Key: `{get_masked_key(gemini_k)}` (Status: Active)")
        model_name = st.selectbox("Gemini Model", [DEFAULT_GEMINI_MODEL, "gemini-flash-latest"], index=0)
    else:
        model_name = "None"
        st.warning("No API keys found in .env. Model generation disabled.")

    st.markdown("---")
    st.markdown("### Source Document Library")
    docs = lib.list_documents()

    if not docs:
        st.info("No documents currently indexed.")
        selected_doc_ids = []
    else:
        st.caption(f"{len(docs)} documents indexed with page provenance.")
        doc_selection_map = {}
        for d in docs:
            is_checked = st.checkbox(
                f"{d['title']}",
                value=True,
                key=f"chk_{d['doc_id']}",
                help=f"Pages: {d['page_count']} | Chunks: {d['chunk_count']}"
            )
            if is_checked:
                doc_selection_map[d["doc_id"]] = d

        selected_doc_ids = list(doc_selection_map.keys())

    # File Ingestion Uploader
    with st.expander("Upload Course Material (PDF / TXT)"):
        uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt", "md"])
        custom_title = st.text_input("Document Title (optional)")
        if uploaded_file and st.button("Index Document", type="secondary"):
            with st.spinner("Extracting text and chunking passages with page metadata..."):
                title_to_use = custom_title.strip() or uploaded_file.name
                file_bytes = uploaded_file.read()
                f_type = "pdf" if uploaded_file.name.lower().endswith(".pdf") else "txt"
                ingest_res = lib.add_document(
                    title=title_to_use,
                    filename=uploaded_file.name,
                    content_bytes_or_str=file_bytes,
                    file_type=f_type
                )
                if ingest_res.get("success"):
                    st.success(ingest_res["message"])
                    time.sleep(1)
                    st.rerun()
                elif ingest_res.get("is_duplicate"):
                    st.warning(ingest_res["message"])
                else:
                    st.error(ingest_res.get("error", "Failed to ingest document"))

    st.markdown("---")
    st.markdown("### Academic Metadata")
    st.markdown("""
    **Vellore Institute of Technology**  
    *Operating Systems Lab (BCSE303P)*  
    - **Ashwin Das** (24BCI0254)  
    - **Rohan Shah** (24BCE2121)  
    *Academic Year: 2026-2027*
    """)


# Header Banner
st.markdown("""
<div class="header-banner">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 class="app-title">OSTutorLLM</h1>
            <p class="app-subtitle">A Retrieval-Grounded Learning Assistant for Operating Systems (BCSE303P)</p>
        </div>
        <div style="text-align: right;">
            <span style="background: rgba(99,102,241,0.2); border: 1px solid #6366f1; color: #a5b4fc; padding: 6px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600;">
                RAG + Deterministic Simulators
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# Main Application Navigation (Clean text without emojis)
nav_choice = st.radio(
    "Application Section",
    ["Tutor", "Study Tools", "OS Lab", "Progress & Revision"],
    horizontal=True,
    label_visibility="collapsed"
)



# Fetch active passages based on sidebar selection
active_passages = lib.get_passages_for_docs(selected_doc_ids if selected_doc_ids else None)


# ============================================================================
# SECTION 1: CONVERSATIONAL TUTOR
# ============================================================================
if nav_choice == "Tutor":
    st.markdown("### Grounded Conversational Tutor")
    st.caption("Ask natural-language questions about Operating Systems. Responses are strictly grounded in your indexed textbook passages with explicit page citations.")

    col_t_ctrl, col_t_chat = st.columns([1, 3])

    with col_t_ctrl:
        st.markdown("#### Tutor Settings")
        ans_length = st.select_slider(
            "Explanation Length",
            options=["Short", "Medium", "Long"],
            value="Medium",
            help="Short: quick revision definition; Medium: concept + example; Long: comprehensive walkthrough."
        )

        st.caption(f"Grounding Source Set: **{len(selected_doc_ids)} document(s)** selected.")
        st.caption(f"Searchable Passages: **{len(active_passages)} chunks**.")

        if st.button("New Chat", use_container_width=True):
            st.session_state.chat_messages = []
            st.rerun()

        st.markdown("---")
        st.markdown("##### Suggested Questions")
        sample_prompts = [
            "Explain the four necessary conditions for deadlock.",
            "What is a page fault and how is it serviced?",
            "Compare FCFS and SJF scheduling algorithms.",
            "How does Dijkstra's Banker's algorithm work?",
            "What is Belady's Anomaly in page replacement?"
        ]
        for sp in sample_prompts:
            if st.button(sp, key=f"btn_p_{sp[:15]}", use_container_width=True):
                st.session_state.pending_prompt = sp
                st.rerun()

    with col_t_chat:
        # Display chat history
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("citations"):
                    st.markdown("**Evidence Citations:**")
                    cit_badges = ""
                    for c in msg["citations"]:
                        cit_badges += f'<span class="citation-badge">Source: {c["doc_name"]} (Page {c["page"]})</span>'
                    st.markdown(cit_badges, unsafe_allow_html=True)

                    with st.expander("View Referenced Textbook Passages"):
                        for c in msg["citations"]:
                            st.markdown(f"**{c['doc_name']} — Page {c['page']}** (Relevance: `{c.get('score', 0)}`)")
                            st.markdown(f'<div class="citation-accordion">"{c["snippet"]}"</div>', unsafe_allow_html=True)
                            st.write("")

                if msg.get("grounding"):
                    g = msg["grounding"]
                    g_pct = g.get("support_percentage", 0)
                    if g_pct < 50:
                        st.caption(f"Grounding confidence: {g_pct}% support. {g.get('warning', '')}")
                    else:
                        st.caption(f"Grounded with {g_pct}% source textual alignment.")

        # Handle User Input
        user_input = st.chat_input("Ask a question about Operating Systems concepts, algorithms, or lab topics...")
        if hasattr(st.session_state, "pending_prompt") and st.session_state.pending_prompt:
            user_input = st.session_state.pending_prompt
            del st.session_state.pending_prompt

        if user_input:
            st.session_state.chat_messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Retrieving evidence passages and synthesizing grounded response..."):
                    history_payload = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.chat_messages[:-1]
                    ]

                    tutor_res = generate_tutor_response(
                        question=user_input,
                        all_passages=active_passages,
                        conversation_history=history_payload,
                        answer_length=ans_length,
                        provider=selected_provider,
                        top_k=4
                    )

                    if not tutor_res.get("success"):
                        err_msg = f"Generation notice: {tutor_res.get('error', 'Generation failed')}"
                        st.error(err_msg)
                        st.session_state.chat_messages.append({"role": "assistant", "content": err_msg})
                    else:
                        ans_text = tutor_res["answer"]
                        citations = tutor_res["citations"]
                        grounding = tutor_res["grounding"]

                        st.markdown(ans_text)

                        if citations:
                            st.markdown("**Evidence Citations:**")
                            cit_badges = ""
                            for c in citations:
                                cit_badges += f'<span class="citation-badge">Source: {c["doc_name"]} (Page {c["page"]})</span>'
                            st.markdown(cit_badges, unsafe_allow_html=True)

                            with st.expander("View Referenced Textbook Passages"):
                                for c in citations:
                                    st.markdown(f"**{c['doc_name']} — Page {c['page']}** (Relevance: `{c.get('score', 0)}`)")
                                    st.markdown(f'<div class="citation-accordion">"{c["snippet"]}"</div>', unsafe_allow_html=True)
                                    st.write("")

                        g_pct = grounding.get("support_percentage", 0)
                        if g_pct < 50:
                            st.caption(f"Grounding confidence: {g_pct}% support. {grounding.get('warning', '')}")
                        else:
                            st.caption(f"Grounded with {g_pct}% source textual alignment.")

                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": ans_text,
                            "citations": citations,
                            "grounding": grounding
                        })


# ============================================================================
# SECTION 2: STUDY TOOLS (Flashcards, Quiz, Mind Map, Syllabus Coverage)
# ============================================================================
elif nav_choice == "Study Tools":
    st.markdown("### Integrated Study Tools")
    st.caption("Active recall, multiple-choice assessment, concept mind mapping, and syllabus coverage analysis.")

    tool_tab1, tool_tab2, tool_tab3, tool_tab4 = st.tabs([
        "Flashcards",
        "Multiple-Choice Quiz",
        "Concept Mind Map",
        "Syllabus Coverage"
    ])

    # ------------------------------------------------------------------------
    # SUBTAB 1: FLASHCARDS
    # ------------------------------------------------------------------------
    with tool_tab1:
        st.markdown("#### Active Recall Flashcards")
        st.caption("Generates verified front-and-back flashcards with stable IDs to support spaced repetition.")

        fc_c1, fc_c2, fc_c3 = st.columns([2, 1, 1])
        with fc_c1:
            fc_topic = st.text_input("Flashcard Topic", value="Deadlocks and Coffman Conditions", key="fc_topic_input")
        with fc_c2:
            fc_count = st.selectbox("Deck Size", [5, 8, 10], index=0)
        with fc_c3:
            st.write("")
            st.write("")
            gen_fc_btn = st.button("Generate Flashcard Deck", type="primary", use_container_width=True)

        if gen_fc_btn:
            with st.spinner("Retrieving evidence and compiling distinct prompt-answer flashcards..."):
                deck_res = generate_flashcards(
                    topic=fc_topic,
                    passages=active_passages,
                    requested_count=fc_count,
                    provider=selected_provider
                )
                if deck_res.get("success"):
                    st.session_state.fc_deck = deck_res["cards"]
                    st.session_state.fc_index = 0
                    st.session_state.fc_show_back = False
                    st.success(deck_res["message"])
                else:
                    st.error(deck_res.get("error", "Could not generate flashcards"))

        # Flashcard Player UI
        deck = st.session_state.fc_deck
        if deck:
            idx = st.session_state.fc_index
            card = deck[idx]

            st.markdown(f"**Card {idx + 1} of {len(deck)}** — *Citation: {card.get('citation', 'Textbook')}*")

            front_text = card["front"]
            back_text = card["back"]

            if not st.session_state.fc_show_back:
                st.markdown(f"""
                <div class="flashcard-box">
                    <div style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.1em; color: #a5b4fc; margin-bottom: 8px;">Question / Prompt</div>
                    <div class="flashcard-prompt">{front_text}</div>
                    <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 14px;">(Click 'Reveal Answer' below to check recall)</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="flashcard-box">
                    <div style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.1em; color: #a5b4fc; margin-bottom: 8px;">Question / Prompt</div>
                    <div class="flashcard-prompt">{front_text}</div>
                    <div class="flashcard-answer"><strong>Answer:</strong> {back_text}</div>
                </div>
                """, unsafe_allow_html=True)

            b_col1, b_col2, b_col3, b_col4 = st.columns([1, 1, 1, 1])
            with b_col1:
                if st.button("Previous Card", disabled=(idx == 0), use_container_width=True):
                    st.session_state.fc_index -= 1
                    st.session_state.fc_show_back = False
                    st.rerun()

            with b_col2:
                flip_label = "Hide Answer" if st.session_state.fc_show_back else "Reveal Answer"
                if st.button(flip_label, use_container_width=True):
                    st.session_state.fc_show_back = not st.session_state.fc_show_back
                    st.rerun()

            with b_col3:
                if st.button("Mark as Missed", use_container_width=True):
                    tracker.record_flashcard_review(card["id"], front_text, fc_topic, is_mastered=False)
                    st.toast("Saved to revision queue: Prioritizing this concept in Progress and Revision.")
                    if idx < len(deck) - 1:
                        st.session_state.fc_index += 1
                        st.session_state.fc_show_back = False
                    st.rerun()

            with b_col4:
                if st.button("Mark as Mastered", type="primary", use_container_width=True):
                    tracker.record_flashcard_review(card["id"], front_text, fc_topic, is_mastered=True)
                    st.toast("Card marked as mastered. Added to progress records.")
                    if idx < len(deck) - 1:
                        st.session_state.fc_index += 1
                        st.session_state.fc_show_back = False
                    st.rerun()

    # ------------------------------------------------------------------------
    # SUBTAB 2: QUIZ
    # ------------------------------------------------------------------------
    with tool_tab2:
        st.markdown("#### Multiple-Choice Assessment")
        st.caption("Generates rigorous 4-option multiple-choice items with strict answer mapping and source explanations.")

        q_c1, q_c2, q_c3 = st.columns([2, 1, 1])
        with q_c1:
            quiz_topic = st.text_input("Quiz Topic", value="CPU Scheduling and Criteria", key="quiz_topic_input")
        with q_c2:
            quiz_num = st.selectbox("Number of Questions", [3, 5, 8], index=1)
        with q_c3:
            st.write("")
            st.write("")
            gen_quiz_btn = st.button("Generate Quiz", type="primary", use_container_width=True)

        if gen_quiz_btn:
            with st.spinner("Generating validated multiple-choice questions with verified distractors..."):
                q_res = generate_quiz(
                    topic=quiz_topic,
                    passages=active_passages,
                    num_questions=quiz_num,
                    provider=selected_provider
                )
                if q_res.get("success"):
                    st.session_state.quiz_data = q_res["questions"]
                    st.session_state.quiz_submitted = False
                    st.session_state.quiz_selections = {}
                    st.success(q_res["message"])
                else:
                    st.error(q_res.get("error", "Could not generate quiz"))

        questions = st.session_state.quiz_data
        if questions:
            st.markdown("---")
            score = 0
            all_answered = True

            for q_idx, q in enumerate(questions):
                st.markdown(f"##### Question {q_idx + 1}: {q['question']}")
                opts = q["options"]
                user_choice = st.radio(
                    f"Select your answer for Question {q_idx + 1}:",
                    options=opts,
                    key=f"quiz_radio_{q['id']}",
                    index=None if not st.session_state.quiz_submitted and q["id"] not in st.session_state.quiz_selections else (
                        opts.index(st.session_state.quiz_selections[q["id"]]) if q["id"] in st.session_state.quiz_selections and st.session_state.quiz_selections[q["id"]] in opts else None
                    )
                )

                if user_choice:
                    st.session_state.quiz_selections[q["id"]] = user_choice
                else:
                    all_answered = False

                if st.session_state.quiz_submitted:
                    chosen = st.session_state.quiz_selections.get(q["id"])
                    is_correct = (chosen == q["correct_answer"])
                    if is_correct:
                        score += 1
                        st.success(f"[CORRECT] {q['correct_answer']}")
                    else:
                        st.error(f"[INCORRECT] Your answer: {chosen or 'None'} | Correct answer: {q['correct_answer']}")

                    st.info(f"**Explanation:** {q['explanation']}\n\n*Reference:* {q.get('citation', '')}")

                st.write("")

            st.markdown("---")
            if not st.session_state.quiz_submitted:
                if st.button("Submit Quiz and Check Results", type="primary"):
                    st.session_state.quiz_submitted = True
                    for q in questions:
                        chosen = st.session_state.quiz_selections.get(q["id"], "")
                        is_corr = (chosen == q["correct_answer"])
                        tracker.record_quiz_attempt(
                            question_id=q["id"],
                            question=q["question"],
                            topic=quiz_topic,
                            selected_option=chosen,
                            correct_answer=q["correct_answer"],
                            is_correct=is_corr
                        )
                    st.rerun()
            else:
                final_pct = round((score / len(questions)) * 100, 1)
                st.markdown(f"### Quiz Completed: Score {score} / {len(questions)} ({final_pct}%)")
                if st.button("Retake or Generate New Quiz"):
                    st.session_state.quiz_submitted = False
                    st.session_state.quiz_selections = {}
                    st.rerun()

    # ------------------------------------------------------------------------
    # SUBTAB 3: MIND MAP
    # ------------------------------------------------------------------------
    with tool_tab3:
        st.markdown("#### Concept Mind Map")
        st.caption("Visualizes topic concepts, algorithms, and conditions. Distinguishes source-supported relationships from inferred organizational links.")

        mm_c1, mm_c2 = st.columns([3, 1])
        with mm_c1:
            mm_topic = st.text_input("Mind Map Topic", value="Deadlocks, Prevention, and Avoidance", key="mm_topic_input")
        with mm_c2:
            st.write("")
            st.write("")
            gen_mm_btn = st.button("Generate Concept Map", type="primary", use_container_width=True)

        if gen_mm_btn:
            with st.spinner("Analyzing conceptual relationships and generating SVG graph geometry..."):
                mm_res = generate_mind_map(
                    topic=mm_topic,
                    passages=active_passages,
                    provider=selected_provider
                )
                st.session_state.mind_map_res = mm_res

        mm_data = st.session_state.mind_map_res
        if mm_data and mm_data.get("success"):
            st.markdown("---")

            # Metrics row
            br_col1, br_col2, br_col3 = st.columns(3)
            with br_col1:
                st.metric("Total Concept Nodes", len(mm_data.get("nodes", [])))
            with br_col2:
                st.metric("Source-Supported Relationships", mm_data.get("supported_edge_count", 0), help="Directly supported by textbook citations")
            with br_col3:
                st.metric("Inferred Groupings", mm_data.get("inferred_edge_count", 0), help="Organizational hierarchy inferred for navigation")

            mm_view_tab1, mm_view_tab2 = st.tabs(["Interactive Diagram View", "Structured Concept Catalog"])

            with mm_view_tab1:
                st.markdown(f"##### Concept Graph for '{mm_data['topic']}'")
                # Render SVG natively inside iframe via Streamlit HTML component
                components.html(mm_data["svg"], height=620, scrolling=True)

            with mm_view_tab2:
                st.markdown("##### Node Catalog and Citations")
                catalog_rows = []
                for n in mm_data["nodes"]:
                    catalog_rows.append({
                        "Node Label": n["label"],
                        "Category": n.get("category", "concept"),
                        "Description": n.get("description", "")
                    })
                st.dataframe(pd.DataFrame(catalog_rows), use_container_width=True, hide_index=True)

                st.markdown("##### Labeled Relationships")
                edge_rows = []
                for e in mm_data["edges"]:
                    edge_rows.append({
                        "Source Concept": e["from"],
                        "Target Concept": e["to"],
                        "Relationship": e["label"],
                        "Status": "Source-Supported" if e.get("status") == "source_supported" else "Inferred",
                        "Citation": e.get("citation", "Textbook")
                    })
                st.dataframe(pd.DataFrame(edge_rows), use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------------
    # SUBTAB 4: SYLLABUS COVERAGE
    # ------------------------------------------------------------------------
    with tool_tab4:
        st.markdown("#### Syllabus Coverage Analysis")
        st.caption("Compares course syllabus topics against indexed textbooks. Excludes the syllabus itself from evidence to prevent circular reasoning.")

        cov_col1, cov_col2 = st.columns([1, 2])
        with cov_col1:
            st.markdown("##### Syllabus Definition")
            use_default_syl = st.checkbox("Use Official BCSE303P Syllabus Topics", value=True)
            run_cov_btn = st.button("Analyze Syllabus Coverage", type="primary", use_container_width=True)

        with cov_col2:
            if use_default_syl:
                st.markdown("**Official BCSE303P Core Topics to Analyze:**")
                for s in DEFAULT_OS_SYLLABUS:
                    st.markdown(f"- **{s['topic']}**: {', '.join(s['subtopics'])}")
                active_syllabus = DEFAULT_OS_SYLLABUS
            else:
                st.markdown("**Custom Syllabus Mode:** Enter topics and subtopics below (Format: `Topic Name: Subtopic 1, Subtopic 2, ...` or one per line)")
                custom_input = st.text_area(
                    "Custom Syllabus Topics",
                    value=st.session_state.custom_syllabus_text,
                    height=160,
                    help="Enter one topic per line with colon-separated subtopics."
                )
                st.session_state.custom_syllabus_text = custom_input
                parsed_custom = parse_custom_syllabus_text(custom_input)

                if parsed_custom:
                    st.caption(f"Parsed {len(parsed_custom)} custom topic entries:")
                    for pt in parsed_custom:
                        st.markdown(f"- **{pt['topic']}**: {', '.join(pt['subtopics'])}")
                    active_syllabus = parsed_custom
                else:
                    st.warning("Please enter at least one valid topic.")
                    active_syllabus = DEFAULT_OS_SYLLABUS

        if run_cov_btn:
            with st.spinner("Scanning indexed course materials and cross-verifying subtopics..."):
                cov_eval = evaluate_syllabus_coverage(
                    syllabus_topics=active_syllabus,
                    evidence_passages=active_passages
                )
                st.session_state.coverage_res = cov_eval

        c_data = st.session_state.coverage_res
        if c_data:
            st.markdown("---")
            st.markdown("##### Document Coverage Results")
            cm1, cm2, cm3, cm4 = st.columns(4)
            cm1.metric("Overall Document Coverage", f"{c_data['overall_coverage_percentage']}%")
            cm2.metric("Covered Topics", f"{c_data['covered_count']} / {c_data['total_topics']}")
            cm3.metric("Partially Covered", f"{c_data['partially_covered_count']}")
            cm4.metric("Uncovered Topics", f"{c_data['not_covered_count']}")

            for t_rep in c_data["topics_report"]:
                stat = t_rep["status"]
                badge = f"[{stat.upper()}]"
                with st.expander(f"{badge} — {t_rep['topic']} ({t_rep['coverage_percentage']}% Subtopics)"):
                    st.markdown(f"**Assessment:** {t_rep['explanation']}")
                    st.markdown("**Subtopic Breakdown:**")
                    for sub in t_rep["subtopics"]:
                        s_icon = "[Covered]" if sub["is_covered"] else "[Missing]"
                        st.markdown(f"- {s_icon} **{sub['subtopic']}** (Relevance Score: `{sub['score']}`)")

                    if t_rep["citations"]:
                        st.markdown("**Textbook Evidence:**")
                        for cit in t_rep["citations"]:
                            st.caption(f"Source: {cit['doc_name']} (Page {cit['page']}): \"{cit['snippet']}\"")


# ============================================================================
# SECTION 3: OPERATING SYSTEMS LABORATORY SIMULATORS
# ============================================================================
elif nav_choice == "OS Lab":
    render_os_lab()


# ============================================================================
# SECTION 4: PROGRESS & REVISION
# ============================================================================
elif nav_choice == "Progress & Revision":
    st.markdown("### Progress Tracking and Targeted Revision")
    st.caption("Tracks flashcard mastery, quiz performance per topic, and generates explainable revision recommendations.")

    p_summary = tracker.get_progress_summary()

    pm1, pm2, pm3, pm4 = st.columns(4)
    pm1.metric("Flashcards Practiced", p_summary["total_cards_reviewed"])
    pm2.metric("Card Mastery Rate", f"{p_summary['card_mastery_rate']}%", help="Percentage of cards marked as mastered")
    pm3.metric("Quiz Questions Answered", p_summary["total_quiz_attempts"])
    pm4.metric("Overall Quiz Accuracy", f"{p_summary['quiz_accuracy_rate']}%")

    col_rec, col_top = st.columns([1, 1])

    with col_rec:
        st.markdown("#### Targeted Revision Queue")
        st.caption("Prioritizes concepts with repeated misses or low quiz accuracy.")
        recs = tracker.get_targeted_revision_recommendations()

        if not recs:
            st.info("No overdue or weak items in the revision queue. Solid mastery across practiced topics.")
        else:
            for r in recs[:6]:
                p_badge = f"[{r['priority'].upper()} PRIORITY]"
                with st.container():
                    st.markdown(f"""
                    <div class="glass-card">
                        <span style="font-size: 0.75rem; background: rgba(239, 68, 68, 0.2); color: #f87171; padding: 2px 8px; border-radius: 4px; font-weight: 700;">{p_badge}</span>
                        <div style="font-weight: 700; font-size: 1.05rem; margin-top: 6px; color: #f1f5f9;">{r['title']}</div>
                        <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 4px;">{r['reason']}</div>
                        <div style="font-size: 0.8rem; color: #38bdf8; margin-top: 4px;">Topic: {r['topic']}</div>
                    </div>
                    """, unsafe_allow_html=True)

    with col_top:
        st.markdown("#### Accuracy by Topic")
        tb = p_summary["topic_breakdown"]
        if not tb:
            st.info("No topic activity recorded yet. Take a quiz or review flashcards to see topic breakdowns.")
        else:
            t_rows = []
            for t_name, t_data in tb.items():
                t_rows.append({
                    "Topic": t_name,
                    "Attempts": t_data["total_attempts"],
                    "Correct": t_data["correct"],
                    "Accuracy": f"{t_data['accuracy']}%",
                    "Status": t_data["status"]
                })
            st.dataframe(pd.DataFrame(t_rows), use_container_width=True, hide_index=True)

        st.markdown("---")
        if st.button("Reset All Progress Records"):
            tracker.reset_progress()
            st.success("Progress records cleared.")
            st.rerun()
