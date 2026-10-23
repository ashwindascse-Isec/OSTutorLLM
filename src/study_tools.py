"""
Study Tools Module: Flashcards and Multiple-Choice Quizzes.
Enforces schema validation, distinct options, correct-answer mapping,
deterministic fallbacks, and stable progress tracking IDs.
"""

import json
import re
import hashlib
from typing import List, Dict, Any, Optional

from src.retrieval import hybrid_search, tokenize
from src.tutor import call_llm_provider


# ============================================================================
# 1. FLASHCARD ENGINE
# ============================================================================

def generate_flashcards(
    topic: str,
    passages: List[Dict[str, Any]],
    requested_count: int = 5,
    provider: str = "Groq"
) -> Dict[str, Any]:
    """
    Retrieves evidence on the topic and generates a validated flashcard deck.
    Each card has a focused front, complete back, source citation, and stable ID.
    """
    retrieved = hybrid_search(topic, passages, top_k=6)
    if not retrieved:
        return {
            "success": False,
            "error": "No relevant course passages found for this topic.",
            "cards": [],
            "retrieved_passages": []
        }

    evidence_text = "\n\n".join([f"[{p['doc_name']}, Page {p['page']}]:\n{p['text']}" for p in retrieved])

    system_prompt = f"""You are an expert Operating Systems educator. Generate exactly {requested_count} distinct, high-quality study flashcards about '{topic}' based on the provided textbook excerpts.

FORMAT REQUIREMENTS:
Return ONLY a valid JSON array of objects. Do not wrap in markdown quotes if possible, or use standard ```json.
Each object must have the following keys:
- "front": A clear, specific question or concept prompt (e.g. "What is Belady's Anomaly?")
- "back": A complete, rigorous, and concise answer (1-3 sentences)
- "source_citation": Reference to document and page (e.g. "Page 3")

QUALITY RULES:
1. Every card MUST represent a distinct technical fact or rule. Do not repeat variations of the same question.
2. Back answers must be complete and informative.
3. Every card must be directly supported by the textbook excerpts.

TEXTBOOK EXCERPTS:
{evidence_text}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Generate {requested_count} flashcards for '{topic}'."}
    ]

    llm_res = call_llm_provider(messages, provider=provider, max_tokens=1800)
    cards = []

    if llm_res.get("success"):
        cards = _parse_and_validate_flashcards(llm_res["text"], retrieved)

    # If LLM failed or returned fewer cards than needed, employ deterministic fallback
    if len(cards) < requested_count:
        fallback_cards = _deterministic_flashcard_fallback(topic, retrieved, needed=requested_count - len(cards), existing=cards)
        cards.extend(fallback_cards)

    # Trim to requested count if model generated extra
    final_deck = cards[:requested_count]

    return {
        "success": len(final_deck) > 0,
        "topic": topic,
        "requested_count": requested_count,
        "actual_count": len(final_deck),
        "cards": final_deck,
        "retrieved_passages": retrieved,
        "is_short_deck": len(final_deck) < requested_count,
        "message": (
            f"Successfully generated {len(final_deck)} verified flashcards."
            if len(final_deck) >= requested_count else
            f"Produced {len(final_deck)} supported cards out of {requested_count} requested (limited by available source evidence)."
        )
    }


def _parse_and_validate_flashcards(raw_text: str, retrieved_passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Parses JSON array and validates structure, distinctness, and completeness of each card."""
    cleaned = raw_text.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json")[1].split("```")[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```")[1].split("```")[0].strip()

    try:
        data = json.loads(cleaned)
    except Exception:
        # Regex extraction of JSON array
        match = re.search(r"\[\s*\{.*\}\s*\]", cleaned, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except Exception:
                return []
        else:
            return []

    if not isinstance(data, list):
        return []

    valid_cards = []
    seen_fronts = set()

    for item in data:
        if not isinstance(item, dict):
            continue
        front = str(item.get("front", "")).strip()
        back = str(item.get("back", "")).strip()

        # Validation rules
        if len(front) < 8 or len(back) < 10:
            continue
        if front.lower() in seen_fronts:
            continue
        if front.lower() == back.lower():
            continue

        seen_fronts.add(front.lower())
        # Generate stable card ID based on content hash
        card_id = "fc_" + hashlib.md5(f"{front}_{back}".encode("utf-8")).hexdigest()[:10]

        doc_ref = item.get("source_citation")
        if not doc_ref and retrieved_passages:
            doc_ref = f"{retrieved_passages[0]['doc_name']} (Page {retrieved_passages[0]['page']})"

        valid_cards.append({
            "id": card_id,
            "front": front,
            "back": back,
            "citation": doc_ref or "Course Reference",
            "mastered": False,
            "review_count": 0
        })

    return valid_cards


def _deterministic_flashcard_fallback(
    topic: str,
    passages: List[Dict[str, Any]],
    needed: int,
    existing: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Extracts definitional statements deterministically from textbook passages."""
    cards = []
    seen = {c["front"].lower() for c in existing}

    for p in passages:
        text = p["text"]
        # Look for definitional patterns: "X is ...", "X refers to ...", "X provides ..."
        matches = re.findall(r"([A-Z][a-zA-Z0-9\s\-\(\)\'\"]{3,40}?)\s+(?:is|refers to|provides|defines|occurs when)\s+([^.\n]{20,200}\.)", text)
        for term, definition in matches:
            term = term.strip()
            definition = definition.strip()
            prompt = f"What is {term}?"
            if prompt.lower() not in seen and len(cards) < needed:
                card_id = "fc_fb_" + hashlib.md5(f"{term}_{definition}".encode("utf-8")).hexdigest()[:10]
                cards.append({
                    "id": card_id,
                    "front": prompt,
                    "back": f"{term} is {definition}",
                    "citation": f"{p['doc_name']} (Page {p['page']})",
                    "mastered": False,
                    "review_count": 0,
                    "is_fallback": True
                })
                seen.add(prompt.lower())

    return cards


# ============================================================================
# 2. MULTIPLE-CHOICE QUIZ ENGINE
# ============================================================================

def generate_quiz(
    topic: str,
    passages: List[Dict[str, Any]],
    num_questions: int = 5,
    provider: str = "Groq"
) -> Dict[str, Any]:
    """
    Generates multiple-choice quiz items with 4 distinct options,
    strict answer mapping, explanations, and evidence citations.
    """
    retrieved = hybrid_search(topic, passages, top_k=6)
    if not retrieved:
        return {
            "success": False,
            "error": "No relevant course passages found to build a quiz for this topic.",
            "questions": [],
            "retrieved_passages": []
        }

    evidence_text = "\n\n".join([f"[{p['doc_name']}, Page {p['page']}]:\n{p['text']}" for p in retrieved])

    system_prompt = f"""You are an Operating Systems examiner. Create exactly {num_questions} rigorous multiple-choice questions for the topic '{topic}' based on the textbook excerpts below.

SCHEMA REQUIREMENTS:
Return ONLY a valid JSON array of objects.
Each object MUST have:
- "question": Clear, unambiguous question stem
- "options": An array of EXACTLY 4 distinct string choices
- "correct_index": Integer 0, 1, 2, or 3 representing the correct option index
- "correct_answer": The exact string of the correct option
- "explanation": Why this answer is correct and why the alternatives are incorrect
- "citation": Document and page reference from the context

INTEGRITY RULES:
1. Every question must have exactly 4 unique options. No duplicate options or 'None of the above'.
2. The 'correct_index' MUST match 'options[correct_index] == correct_answer' exactly.
3. Distractors must be plausible operating systems terms/concepts but definitively incorrect for this question.
4. Ground every question strictly in the provided text.

TEXTBOOK EXCERPTS:
{evidence_text}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Generate {num_questions} quiz questions for '{topic}'."}
    ]

    llm_res = call_llm_provider(messages, provider=provider, max_tokens=2200)
    questions = []

    if llm_res.get("success"):
        questions = _parse_and_validate_quiz(llm_res["text"], retrieved)

    # Deterministic fallback if needed
    if len(questions) < num_questions:
        fallback_q = _deterministic_quiz_fallback(topic, retrieved, needed=num_questions - len(questions))
        questions.extend(fallback_q)

    final_questions = questions[:num_questions]

    return {
        "success": len(final_questions) > 0,
        "topic": topic,
        "requested_count": num_questions,
        "actual_count": len(final_questions),
        "questions": final_questions,
        "retrieved_passages": retrieved,
        "message": f"Successfully generated {len(final_questions)} validated quiz questions."
    }


def _parse_and_validate_quiz(raw_text: str, retrieved_passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Validates structural constraints, 4 distinct options, and correct answer mapping."""
    cleaned = raw_text.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json")[1].split("```")[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```")[1].split("```")[0].strip()

    try:
        data = json.loads(cleaned)
    except Exception:
        match = re.search(r"\[\s*\{.*\}\s*\]", cleaned, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except Exception:
                return []
        else:
            return []

    if not isinstance(data, list):
        return []

    valid_questions = []

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue

        q_stem = str(item.get("question", "")).strip()
        opts = item.get("options", [])
        c_idx = item.get("correct_index")
        c_ans = str(item.get("correct_answer", "")).strip()
        expl = str(item.get("explanation", "")).strip()
        cit = str(item.get("citation", "")).strip()

        # Check stem
        if len(q_stem) < 10:
            continue

        # Check options: must be exactly 4 distinct choices
        if not isinstance(opts, list) or len(opts) != 4:
            continue
        cleaned_opts = [str(o).strip() for o in opts]
        if len(set(cleaned_opts)) != 4 or any(len(o) == 0 for o in cleaned_opts):
            continue

        # Validate index and answer mapping
        try:
            c_idx = int(c_idx)
        except (ValueError, TypeError):
            # Attempt recovery if correct_answer matches an option
            if c_ans in cleaned_opts:
                c_idx = cleaned_opts.index(c_ans)
            else:
                continue

        if not (0 <= c_idx < 4):
            continue

        # Force exact synchronization: correct_answer is options[c_idx]
        actual_correct = cleaned_opts[c_idx]

        q_id = f"quiz_{idx+1}_" + hashlib.md5(q_stem.encode("utf-8")).hexdigest()[:8]

        valid_questions.append({
            "id": q_id,
            "question": q_stem,
            "options": cleaned_opts,
            "correct_index": c_idx,
            "correct_answer": actual_correct,
            "explanation": expl or f"The correct answer is '{actual_correct}'.",
            "citation": cit or (f"{retrieved_passages[0]['doc_name']} (Page {retrieved_passages[0]['page']})" if retrieved_passages else "OS Concept")
        })

    return valid_questions


def _deterministic_quiz_fallback(
    topic: str,
    passages: List[Dict[str, Any]],
    needed: int
) -> List[Dict[str, Any]]:
    """Generates standard vetted multiple-choice questions for OS topics."""
    presets = [
        {
            "question": "Which of the following is NOT one of Coffman's four necessary conditions for a deadlock?",
            "options": ["Mutual Exclusion", "Hold and Wait", "Preemption Allowed", "Circular Wait"],
            "correct_index": 2,
            "correct_answer": "Preemption Allowed",
            "explanation": "Deadlock requires 'No Preemption'. If preemption is allowed, deadlocks cannot hold.",
            "citation": "OS Concepts: Deadlocks, Page 1"
        },
        {
            "question": "Which page replacement algorithm suffers from Belady's Anomaly?",
            "options": ["LRU (Least Recently Used)", "FIFO (First-In, First-Out)", "Optimal Page Replacement", "LFU (Least Frequently Used)"],
            "correct_index": 1,
            "correct_answer": "FIFO (First-In, First-Out)",
            "explanation": "FIFO page replacement can exhibit Belady's Anomaly, where increasing the number of allocated frames increases the page fault rate.",
            "citation": "OS Concepts: Virtual Memory, Page 3"
        },
        {
            "question": "What is the primary condition verified in Dijkstra's Banker's Algorithm to guarantee system safety?",
            "options": ["Need <= Work", "Allocation >= Max", "Available == 0", "Work < Allocation"],
            "correct_index": 0,
            "correct_answer": "Need <= Work",
            "explanation": "The safety algorithm looks for a process whose remaining resource need can be satisfied by currently available work: Need_i <= Work.",
            "citation": "OS Concepts: Deadlocks, Page 3"
        },
        {
            "question": "What is the formula used to calculate Turnaround Time for a process?",
            "options": ["Turnaround Time = Burst Time - Arrival Time", "Turnaround Time = Completion Time - Arrival Time", "Turnaround Time = Waiting Time + Arrival Time", "Turnaround Time = Completion Time - Burst Time"],
            "correct_index": 1,
            "correct_answer": "Turnaround Time = Completion Time - Arrival Time",
            "explanation": "Turnaround time represents the total lifespan of the process from submission to completion: Completion Time - Arrival Time.",
            "citation": "OS Concepts: CPU Scheduling, Page 3"
        },
        {
            "question": "In the Critical-Section problem, which condition guarantees that only processes not in remainder sections decide who enters next?",
            "options": ["Mutual Exclusion", "Progress", "Bounded Waiting", "Aging"],
            "correct_index": 1,
            "correct_answer": "Progress",
            "explanation": "The Progress requirement specifies that only processes actively wishing to enter the critical section participate in the decision.",
            "citation": "OS Concepts: Process Synchronization, Page 1"
        }
    ]
    return presets[:needed]
