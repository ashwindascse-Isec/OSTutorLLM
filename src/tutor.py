"""
Conversational Tutor with Grounded RAG Generation and Multi-Provider LLM Integration.
Supports Groq and Google Gemini, answer length controls, and bounded conversational context.
"""

import os
import json
from typing import List, Dict, Any, Optional

from src.config import (
    get_groq_key,
    get_gemini_key,
    DEFAULT_GROQ_MODEL,
    FALLBACK_GROQ_MODELS,
    DEFAULT_GEMINI_MODEL,
    FALLBACK_GEMINI_MODELS
)
from src.retrieval import hybrid_search
from src.citation_validator import validate_citations, check_answer_grounding


def call_llm_provider(
    messages: List[Dict[str, str]],
    provider: str = "Groq",
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 1500
) -> Dict[str, Any]:
    """
    Unified LLM caller that dispatches to Groq or Google Gemini.
    Features automatic fallback between models and providers on failure.
    """
    provider_clean = provider.strip().capitalize()

    # Try Groq first if selected
    if provider_clean == "Groq":
        groq_key = get_groq_key()
        if not groq_key:
            return {"success": False, "error": "GROQ_API_KEY is not configured in .env."}
        try:
            from groq import Groq
            client = Groq(api_key=groq_key)
            target_model = model or DEFAULT_GROQ_MODEL
            try:
                resp = client.chat.completions.create(
                    model=target_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                text = resp.choices[0].message.content or ""
                return {"success": True, "text": text.strip(), "provider": "Groq", "model": target_model}
            except Exception as model_err:
                # Try fallbacks
                for fb_m in FALLBACK_GROQ_MODELS:
                    try:
                        resp = client.chat.completions.create(
                            model=fb_m,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens
                        )
                        text = resp.choices[0].message.content or ""
                        return {"success": True, "text": text.strip(), "provider": "Groq", "model": fb_m}
                    except Exception:
                        continue
                raise model_err
        except Exception as e:
            # Check if Gemini can be a fallback
            gemini_key = get_gemini_key()
            if gemini_key:
                return _call_gemini_fallback(messages, temperature, max_tokens)
            return {"success": False, "error": f"Groq API Error: {str(e)}"}

    # Gemini provider
    elif provider_clean in ["Gemini", "Google"]:
        gemini_key = get_gemini_key()
        if not gemini_key:
            return {"success": False, "error": "GEMINI_API_KEY / GOOGLE_API_KEY is not configured in .env."}
        try:
            from google import genai
            client = genai.Client(api_key=gemini_key)
            target_model = model or DEFAULT_GEMINI_MODEL

            # Convert messages to prompt string
            prompt_parts = []
            for m in messages:
                role = m.get("role", "user").capitalize()
                prompt_parts.append(f"{role}:\n{m.get('content', '')}")
            full_prompt = "\n\n".join(prompt_parts)

            try:
                resp = client.models.generate_content(
                    model=target_model,
                    contents=full_prompt,
                )
                return {"success": True, "text": (resp.text or "").strip(), "provider": "Gemini", "model": target_model}
            except Exception:
                for fb_m in FALLBACK_GEMINI_MODELS:
                    try:
                        resp = client.models.generate_content(model=fb_m, contents=full_prompt)
                        return {"success": True, "text": (resp.text or "").strip(), "provider": "Gemini", "model": fb_m}
                    except Exception:
                        continue
                raise
        except Exception as e:
            # Fallback to Groq if available
            if get_groq_key():
                return call_llm_provider(messages, provider="Groq", temperature=temperature, max_tokens=max_tokens)
            return {"success": False, "error": f"Gemini API Error: {str(e)}"}

    return {"success": False, "error": f"Unknown provider: {provider}"}


def _call_gemini_fallback(messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> Dict[str, Any]:
    gemini_key = get_gemini_key()
    try:
        from google import genai
        client = genai.Client(api_key=gemini_key)
        prompt_parts = [f"{m['role'].upper()}: {m['content']}" for m in messages]
        resp = client.models.generate_content(
            model=DEFAULT_GEMINI_MODEL,
            contents="\n\n".join(prompt_parts)
        )
        return {"success": True, "text": resp.text.strip(), "provider": "Gemini (Fallback)", "model": DEFAULT_GEMINI_MODEL}
    except Exception as e:
        return {"success": False, "error": f"Both primary and fallback providers failed: {e}"}


def build_tutor_prompt(
    question: str,
    retrieved_passages: List[Dict[str, Any]],
    conversation_history: List[Dict[str, str]],
    answer_length: str = "Medium"
) -> List[Dict[str, str]]:
    """
    Constructs the prompt for the OS Tutor enforcing evidence grounding,
    provenance, and length constraints.
    """
    length_instructions = {
        "Short": "Keep the answer concise and high-yield (1 to 2 paragraphs maximum). Focus strictly on core definitions, key rules, and essential criteria for quick revision.",
        "Medium": "Provide a well-structured explanation (2 to 4 paragraphs) with one practical example or scenario illustrating how the concept operates in an Operating System.",
        "Long": "Provide a comprehensive, in-depth tutorial breakdown. Detail the mechanism step-by-step, explain underlying design trade-offs, compare with related concepts, and provide a concrete example."
    }
    length_guide = length_instructions.get(answer_length, length_instructions["Medium"])

    # Format context passages with citations
    context_blocks = []
    for idx, p in enumerate(retrieved_passages, 1):
        context_blocks.append(
            f"--- [Passage {idx}] Source: {p['doc_name']}, Page: {p['page']} ---\n{p['text']}"
        )
    context_text = "\n\n".join(context_blocks) if context_blocks else "NO EVIDENCE RETRIEVED."

    system_instruction = f"""You are OSTutorLLM, an authoritative, rigorous, and student-focused AI learning assistant for the Operating Systems course (BCSE303P).

CRITICAL GROUNDING & ACCURACY RULES:
1. Base your answer primarily on the supplied textbook passages below.
2. If the passages contain the answer, explain the concepts clearly, accurately, and authoritatively.
3. If the supplied passages DO NOT contain sufficient information to answer the question accurately, explicitly state:
   "The selected course materials do not provide sufficient evidence to answer this question comprehensively." Then provide a cautious, standard Operating Systems explanation while explicitly clarifying that it is outside the selected textbook evidence.
4. When stating facts or definitions that originate from a specific passage, reference the citation as [Source: <doc_name>, Page: <page>].
5. Never invent or hallucinate page numbers, formulas, or algorithmic traces.
6. Length Constraint: {length_guide}

RETRIEVED COURSE EVIDENCE:
{context_text}
"""

    messages = [{"role": "system", "content": system_instruction}]

    # Include bounded conversation history (last 4 turns)
    bounded_history = conversation_history[-4:] if len(conversation_history) > 4 else conversation_history
    for msg in bounded_history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": question})
    return messages


def generate_tutor_response(
    question: str,
    all_passages: List[Dict[str, Any]],
    conversation_history: List[Dict[str, str]],
    answer_length: str = "Medium",
    provider: str = "Groq",
    top_k: int = 4
) -> Dict[str, Any]:
    """
    End-to-end Tutor workflow:
    1. Hybrid retrieval from indexed passages.
    2. Prompt construction with bounded history and evidence.
    3. Generation via configured provider.
    4. Citation validation and grounding score verification.
    """
    retrieved = hybrid_search(question, all_passages, top_k=top_k)

    # Check evidence sufficiency
    evidence_found = len(retrieved) > 0 and (retrieved[0].get("score", 0) >= 0.15)

    messages = build_tutor_prompt(
        question=question,
        retrieved_passages=retrieved,
        conversation_history=conversation_history,
        answer_length=answer_length
    )

    llm_res = call_llm_provider(messages, provider=provider)

    if not llm_res.get("success"):
        return {
            "success": False,
            "error": llm_res.get("error", "Unknown generation error"),
            "retrieved_passages": retrieved,
            "citations": []
        }

    answer_text = llm_res["text"]

    # Validate citations and grounding
    citations_data = validate_citations(retrieved, retrieved)
    grounding_data = check_answer_grounding(answer_text, retrieved)

    return {
        "success": True,
        "answer": answer_text,
        "provider": llm_res.get("provider"),
        "model": llm_res.get("model"),
        "retrieved_passages": retrieved,
        "citations": citations_data["valid_citations"],
        "grounding": grounding_data,
        "evidence_sufficient": evidence_found
    }
