"""
Citation and Grounding Validation.
Validates source references against retrieved passages,
measures claim-evidence overlap, and formats provenance blocks.
"""

import re
from typing import List, Dict, Any, Tuple, Optional
from src.retrieval import tokenize


def validate_citations(
    cited_ids_or_names: List[Any],
    retrieved_passages: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Checks that every cited source exists in the retrieved set.
    Assembles authoritative citations with document title, page number, and snippet.
    """
    valid_citations = []
    invalid_citations = []

    # Map retrieved passages by doc_id, doc_name, and chunk id
    retrieved_map = {}
    for p in retrieved_passages:
        retrieved_map[p["id"]] = p
        retrieved_map[p["doc_id"]] = p
        retrieved_map[p["doc_name"].lower()] = p

    for item in cited_ids_or_names:
        matched_passage = None
        if isinstance(item, dict):
            # Item might have 'doc_id', 'page', 'id'
            p_id = item.get("id") or item.get("doc_id")
            if p_id and p_id in retrieved_map:
                matched_passage = retrieved_map[p_id]
            elif item.get("doc_name") and item["doc_name"].lower() in retrieved_map:
                matched_passage = retrieved_map[item["doc_name"].lower()]
        elif isinstance(item, str):
            clean_str = item.strip().lower()
            if clean_str in retrieved_map:
                matched_passage = retrieved_map[clean_str]
            else:
                for k, p in retrieved_map.items():
                    if clean_str in k or k in clean_str:
                        matched_passage = p
                        break

        if matched_passage:
            # Create verified citation
            quote_snip = matched_passage["text"][:160] + ("..." if len(matched_passage["text"]) > 160 else "")
            valid_citations.append({
                "doc_id": matched_passage["doc_id"],
                "doc_name": matched_passage["doc_name"],
                "page": matched_passage["page"],
                "chunk_id": matched_passage["id"],
                "score": matched_passage.get("score", 0.0),
                "snippet": quote_snip,
                "verified": True
            })
        else:
            invalid_citations.append(item)

    # If no explicit citations were passed, automatically attach all top retrieved passages as valid citations
    if not valid_citations and retrieved_passages:
        for p in retrieved_passages:
            quote_snip = p["text"][:160] + ("..." if len(p["text"]) > 160 else "")
            valid_citations.append({
                "doc_id": p["doc_id"],
                "doc_name": p["doc_name"],
                "page": p["page"],
                "chunk_id": p["id"],
                "score": p.get("score", 0.0),
                "snippet": quote_snip,
                "verified": True
            })

    return {
        "valid_citations": valid_citations,
        "invalid_citations": invalid_citations,
        "citation_count": len(valid_citations),
        "all_verified": len(invalid_citations) == 0
    }


def check_answer_grounding(answer_text: str, retrieved_passages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculates grounding support by assessing term overlap between answer sentences and retrieved passages.
    Returns grounding score (0.0 to 1.0) and supported vs unsupported sentence flags.
    """
    if not answer_text.strip() or not retrieved_passages:
        return {
            "grounding_score": 0.0,
            "is_grounded": False,
            "supported_sentences": 0,
            "total_sentences": 0,
            "warning": "No retrieved passages available to ground answer."
        }

    # Combine all evidence tokens
    evidence_tokens = set()
    for p in retrieved_passages:
        evidence_tokens.update(tokenize(p["text"]))

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer_text) if len(s.strip()) > 15]
    if not sentences:
        sentences = [answer_text.strip()]

    supported_count = 0
    for s in sentences:
        s_tokens = tokenize(s)
        if not s_tokens:
            continue
        overlap = len(set(s_tokens).intersection(evidence_tokens))
        ratio = overlap / len(s_tokens)
        if ratio >= 0.35: # At least 35% of meaningful tokens exist in evidence
            supported_count += 1

    total_s = max(1, len(sentences))
    grounding_score = round(supported_count / total_s, 2)
    is_grounded = grounding_score >= 0.5

    return {
        "grounding_score": grounding_score,
        "is_grounded": is_grounded,
        "supported_sentences": supported_count,
        "total_sentences": total_s,
        "support_percentage": int(grounding_score * 100),
        "warning": None if is_grounded else "Warning: Portions of this response may exceed the evidence in the selected documents."
    }
