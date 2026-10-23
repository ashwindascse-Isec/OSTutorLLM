"""
Source Analysis Module: Mind Maps and Syllabus Coverage.
Implements:
1. Mind-Map Graph Generation with SVG Rendering (source-supported vs inferred edges).
2. Syllabus Coverage Analysis (evaluates topics against textbooks while excluding syllabus as evidence).
"""

import json
import re
import html
import math
from typing import List, Dict, Any, Tuple, Optional

from src.retrieval import hybrid_search, tokenize
from src.tutor import call_llm_provider


# ============================================================================
# 1. MIND MAP ENGINE & SVG RENDERER
# ============================================================================

def generate_mind_map(
    topic: str,
    passages: List[Dict[str, Any]],
    provider: str = "Groq"
) -> Dict[str, Any]:
    """
    Builds a concept mind map from retrieved passages.
    Extracts nodes and labeled relationships, verifying edge endpoints
    and classifying edges into 'source-supported' vs 'inferred'.
    """
    retrieved = hybrid_search(topic, passages, top_k=6)
    if not retrieved:
        return {
            "success": False,
            "error": "No relevant passages found to construct a mind map.",
            "nodes": [],
            "edges": [],
            "svg": ""
        }

    evidence_text = "\n\n".join([f"[{p['doc_name']}, Page {p['page']}]:\n{p['text']}" for p in retrieved])

    system_prompt = f"""You are an Operating Systems concept visualization architect. Create a structured concept mind map for '{topic}' grounded in the provided textbook passages.

SCHEMA:
Return ONLY a valid JSON object with:
- "root_node": "{topic}"
- "nodes": [
    {{"id": "node_id", "label": "Short Title (1-3 words)", "category": "definition|algorithm|condition|component|tradeoff", "description": "1 sentence explanation"}}
  ]
- "edges": [
    {{"from": "node_id_1", "to": "node_id_2", "label": "Relationship (e.g. requires, implements, prevents, causes)", "status": "source_supported|inferred", "citation": "Page X"}}
  ]

RULES:
1. Every edge 'from' and 'to' must reference an existing node ID.
2. Mark edge as 'source_supported' ONLY if the textbook explicitly states this connection. Otherwise mark as 'inferred' (organizational link).
3. Do not create circular self-links (from == to).
4. Provide between 5 and 10 meaningful nodes.

TEXTBOOK PASSAGES:
{evidence_text}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Build concept mind map for '{topic}'."}
    ]

    llm_res = call_llm_provider(messages, provider=provider, max_tokens=1800)
    graph_data = None

    if llm_res.get("success"):
        graph_data = _parse_mind_map_json(llm_res["text"], topic, retrieved)

    # Fallback to deterministic concept graph if LLM generation fails
    if not graph_data or not graph_data.get("nodes"):
        graph_data = _deterministic_mind_map_fallback(topic, retrieved)

    # Render SVG
    svg_content = render_mind_map_svg(graph_data)

    return {
        "success": True,
        "topic": topic,
        "root_node": graph_data.get("root_node", topic),
        "nodes": graph_data.get("nodes", []),
        "edges": graph_data.get("edges", []),
        "svg": svg_content,
        "retrieved_passages": retrieved,
        "supported_edge_count": sum(1 for e in graph_data.get("edges", []) if e.get("status") == "source_supported"),
        "inferred_edge_count": sum(1 for e in graph_data.get("edges", []) if e.get("status") == "inferred")
    }


def _parse_mind_map_json(raw_text: str, root_topic: str, passages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    cleaned = raw_text.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json")[1].split("```")[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```")[1].split("```")[0].strip()

    try:
        data = json.loads(cleaned)
    except Exception:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except Exception:
                return None
        else:
            return None

    if not isinstance(data, dict):
        return None

    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    # Validate nodes
    valid_nodes = []
    node_ids = set()
    root_id = "node_root"
    valid_nodes.append({
        "id": root_id,
        "label": root_topic[:30],
        "category": "root",
        "description": f"Core concept: {root_topic}"
    })
    node_ids.add(root_id)

    for n in nodes:
        if not isinstance(n, dict):
            continue
        n_id = str(n.get("id", "")).strip().replace(" ", "_")
        label = str(n.get("label", "")).strip()[:35]
        cat = str(n.get("category", "concept")).strip()
        desc = str(n.get("description", "")).strip()
        if n_id and label and n_id not in node_ids:
            node_ids.add(n_id)
            valid_nodes.append({
                "id": n_id,
                "label": label,
                "category": cat,
                "description": desc
            })

    # Validate edges
    valid_edges = []
    seen_edge_pairs = set()

    for e in edges:
        if not isinstance(e, dict):
            continue
        f = str(e.get("from", "")).strip().replace(" ", "_")
        t = str(e.get("to", "")).strip().replace(" ", "_")
        lbl = str(e.get("label", "relates to")).strip()[:25]
        status = e.get("status", "inferred")
        cit = str(e.get("citation", "")).strip()

        # Connect orphan root if needed
        if f not in node_ids and f.lower() in root_topic.lower():
            f = root_id
        if t not in node_ids and t.lower() in root_topic.lower():
            t = root_id

        if f in node_ids and t in node_ids and f != t:
            pair = (f, t)
            if pair not in seen_edge_pairs:
                seen_edge_pairs.add(pair)
                valid_edges.append({
                    "from": f,
                    "to": t,
                    "label": lbl,
                    "status": "source_supported" if status == "source_supported" else "inferred",
                    "citation": cit or (f"Page {passages[0]['page']}" if passages else "Textbook")
                })

    # Ensure every non-root node has at least one incoming edge
    connected_targets = {e["to"] for e in valid_edges}.union({e["from"] for e in valid_edges})
    for n in valid_nodes:
        if n["id"] != root_id and n["id"] not in connected_targets:
            valid_edges.append({
                "from": root_id,
                "to": n["id"],
                "label": "involves",
                "status": "inferred",
                "citation": "Inferred grouping"
            })

    return {
        "root_node": root_topic,
        "nodes": valid_nodes,
        "edges": valid_edges
    }


def _deterministic_mind_map_fallback(topic: str, passages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generates structured fallback nodes when LLM parsing is not possible."""
    root_id = "node_root"
    nodes = [{"id": root_id, "label": topic[:25], "category": "root", "description": f"Main topic: {topic}"}]
    edges = []

    # Extract bullet points or section headers
    for idx, p in enumerate(passages[:5], 1):
        lines = p["text"].split("\n")
        for line in lines:
            line = line.strip()
            if (line.startswith(("1.", "2.", "3.", "4.", "-", "*")) or ":" in line) and len(line) < 60:
                clean_lbl = re.sub(r"^[\d\.\-\*\s]+", "", line).split(":")[0].strip()
                if 3 < len(clean_lbl) < 30:
                    n_id = f"node_{idx}_{len(nodes)}"
                    nodes.append({
                        "id": n_id,
                        "label": clean_lbl,
                        "category": "subtopic",
                        "description": f"Discussed on page {p['page']} of {p['doc_name']}."
                    })
                    edges.append({
                        "from": root_id,
                        "to": n_id,
                        "label": "includes",
                        "status": "source_supported",
                        "citation": f"{p['doc_name']} (Page {p['page']})"
                    })
                    if len(nodes) >= 7:
                        break
        if len(nodes) >= 7:
            break

    return {"root_node": topic, "nodes": nodes, "edges": edges}


def render_mind_map_svg(graph_data: Dict[str, Any]) -> str:
    """
    Renders an SVG diagram for the mind map.
    Distinguishes source-supported edges (solid cyan line)
    from inferred relationships (dashed violet line).
    """
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])

    if not nodes:
        return "<svg width='400' height='100'><text x='20' y='50' fill='#aaa'>No nodes to render</text></svg>"

    width = 960
    height = 580
    center_x = width / 2
    center_y = height / 2

    # Layout nodes in radial/orbital pattern around root
    root_node = nodes[0]
    child_nodes = nodes[1:]
    node_coords = {root_node["id"]: (center_x, center_y)}

    n_children = len(child_nodes)
    radius_x = 350
    radius_y = 210

    for i, node in enumerate(child_nodes):
        angle = (2 * math.pi * i / max(1, n_children)) - (math.pi / 2)
        x = center_x + radius_x * math.cos(angle)
        y = center_y + radius_y * math.sin(angle)
        node_coords[node["id"]] = (x, y)

    svg_parts = []
    svg_parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="{height}" style="background-color: #0b0f19; border-radius: 12px; font-family: Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif;">')

    # SVG Definitions: markers for arrows and gradients (zero indent to avoid markdown code block interpretation)
    defs_block = (
        '<defs>'
        '<linearGradient id="rootGrad" x1="0%" y1="0%" x2="100%" y2="100%">'
        '<stop offset="0%" stop-color="#4f46e5"/>'
        '<stop offset="100%" stop-color="#7c3aed"/>'
        '</linearGradient>'
        '<linearGradient id="childGrad" x1="0%" y1="0%" x2="100%" y2="100%">'
        '<stop offset="0%" stop-color="#1e293b"/>'
        '<stop offset="100%" stop-color="#0f172a"/>'
        '</linearGradient>'
        '<marker id="arrow-supported" viewBox="0 0 10 10" refX="28" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#06b6d4"/>'
        '</marker>'
        '<marker id="arrow-inferred" viewBox="0 0 10 10" refX="28" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#a855f7"/>'
        '</marker>'
        '<filter id="glow" x="-20%" y="-20%" width="140%" height="140%">'
        '<feGaussianBlur stdDeviation="3" result="blur"/>'
        '<feComposite in="SourceGraphic" in2="blur" operator="over"/>'
        '</filter>'
        '</defs>'
    )
    svg_parts.append(defs_block)

    # Draw Legend at top
    legend_block = (
        '<g transform="translate(24, 28)">'
        '<line x1="0" y1="10" x2="35" y2="10" stroke="#06b6d4" stroke-width="2.5" marker-end="url(#arrow-supported)"/>'
        '<text x="45" y="14" fill="#38bdf8" font-size="12" font-weight="600">Source-Supported Relationship</text>'
        '<line x1="280" y1="10" x2="315" y2="10" stroke="#a855f7" stroke-width="2" stroke-dasharray="5,4" marker-end="url(#arrow-inferred)"/>'
        '<text x="325" y="14" fill="#c084fc" font-size="12" font-weight="600">Inferred / Organizational Link</text>'
        '</g>'
    )
    svg_parts.append(legend_block)

    # Draw Edges
    for edge in edges:
        f_id = edge["from"]
        t_id = edge["to"]
        if f_id in node_coords and t_id in node_coords:
            x1, y1 = node_coords[f_id]
            x2, y2 = node_coords[t_id]
            is_supported = edge.get("status") == "source_supported"
            stroke_color = "#06b6d4" if is_supported else "#a855f7"
            stroke_width = "2.5" if is_supported else "1.8"
            dash = "" if is_supported else 'stroke-dasharray="6,4"'
            marker = 'marker-end="url(#arrow-supported)"' if is_supported else 'marker-end="url(#arrow-inferred)"'

            svg_parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke_color}" stroke-width="{stroke_width}" {dash} {marker} opacity="0.85"/>')

            # Midpoint edge label
            mx = (x1 + x2) / 2
            my = (y1 + y2) / 2
            lbl = html.escape(edge.get("label", ""))
            if lbl:
                w_lbl = max(60, len(lbl) * 7 + 14)
                svg_parts.append(f'<rect x="{mx - w_lbl/2:.1f}" y="{my - 10:.1f}" width="{w_lbl}" height="20" rx="4" fill="#111827" stroke="#1e293b" stroke-width="1" opacity="0.95"/>')
                svg_parts.append(f'<text x="{mx:.1f}" y="{my + 4:.1f}" text-anchor="middle" fill="{stroke_color}" font-size="10.5" font-weight="600">{lbl}</text>')

    # Draw Nodes
    for node in nodes:
        x, y = node_coords[node["id"]]
        is_root = node["id"] == root_node["id"]
        lbl = html.escape(node.get("label", ""))

        if is_root:
            w, h = 180, 64
            svg_parts.append(f'<rect x="{x - w/2:.1f}" y="{y - h/2:.1f}" width="{w}" height="{h}" rx="14" fill="url(#rootGrad)" stroke="#818cf8" stroke-width="2.5" filter="url(#glow)"/>')
            svg_parts.append(f'<text x="{x:.1f}" y="{y + 6:.1f}" text-anchor="middle" fill="#ffffff" font-size="14.5" font-weight="700">{lbl}</text>')
        else:
            w, h = 160, 52
            svg_parts.append(f'<rect x="{x - w/2:.1f}" y="{y - h/2:.1f}" width="{w}" height="{h}" rx="10" fill="url(#childGrad)" stroke="#334155" stroke-width="1.8"/>')
            svg_parts.append(f'<text x="{x:.1f}" y="{y + 5:.1f}" text-anchor="middle" fill="#e2e8f0" font-size="12" font-weight="600">{lbl}</text>')

    svg_parts.append("</svg>")
    return "".join(svg_parts)



# ============================================================================
# 2. SYLLABUS COVERAGE ENGINE
# ============================================================================

def parse_custom_syllabus_text(raw_text: str) -> List[Dict[str, Any]]:
    """
    Parses custom user-entered syllabus text into structured topics and subtopics.
    Supports:
    - 'Topic: Subtopic 1, Subtopic 2, ...'
    - 'Topic' followed by '- Subtopic' lines
    """
    cleaned = raw_text.strip()
    if not cleaned:
        return []

    topics = []
    lines = [line.strip() for line in cleaned.split("\n") if line.strip()]
    current_topic = None
    current_subs = []

    for line in lines:
        if ":" in line and not line.startswith(("-", "*", "•")):
            # If we had a previous topic, save it
            if current_topic:
                topics.append({"topic": current_topic, "subtopics": current_subs or [current_topic]})
                current_subs = []

            parts = line.split(":", 1)
            t_name = parts[0].strip()
            subs_str = parts[1].strip()
            if subs_str:
                subs = [s.strip() for s in re.split(r"[,;]+", subs_str) if s.strip()]
                topics.append({"topic": t_name, "subtopics": subs or [t_name]})
                current_topic = None
            else:
                current_topic = t_name
        elif line.startswith(("-", "*", "•", "+")):
            sub_name = re.sub(r"^[\-\*\•\+\d\.]+\s*", "", line).strip()
            if sub_name:
                if current_topic:
                    current_subs.append(sub_name)
                else:
                    # Generic topic
                    topics.append({"topic": sub_name, "subtopics": [sub_name]})
        else:
            if current_topic:
                topics.append({"topic": current_topic, "subtopics": current_subs or [current_topic]})
                current_subs = []
            current_topic = line

    if current_topic:
        topics.append({"topic": current_topic, "subtopics": current_subs or [current_topic]})

    return topics


DEFAULT_OS_SYLLABUS = [
    {"topic": "CPU Scheduling Algorithms", "subtopics": ["FCFS Scheduling", "SJF Non-preemptive and Preemptive", "Round Robin Time Quantum", "Priority Scheduling", "Scheduling Criteria"]},
    {"topic": "Deadlocks and Safety", "subtopics": ["Coffman 4 Conditions", "Deadlock Prevention", "Deadlock Avoidance", "Banker's Algorithm", "Resource-Allocation Graph"]},
    {"topic": "Memory Management & Paging", "subtopics": ["Paging and Page Tables", "Demand Paging", "Page Fault Handling", "FIFO Page Replacement", "LRU and Optimal Page Replacement"]},
    {"topic": "Process Synchronization", "subtopics": ["Critical-Section Problem", "Mutual Exclusion", "Semaphores wait and signal", "Mutex Locks", "Classic Synchronization Problems"]},
    {"topic": "File Systems & Mass Storage", "subtopics": ["File Allocation Methods", "Contiguous, Linked, Indexed Allocation", "Disk Scheduling FCFS and SCAN", "Directory Structure"]}
]


def evaluate_syllabus_coverage(
    syllabus_topics: List[Dict[str, Any]],
    evidence_passages: List[Dict[str, Any]],
    excluded_doc_ids: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Evaluates each topic in syllabus against evidence passages.
    Strictly excludes the syllabus document itself from evidence.
    Classifies topics into: Covered, Partially Covered, Not Covered / Insufficient.
    """
    # Filter out excluded documents (e.g. the syllabus document itself)
    active_passages = evidence_passages
    if excluded_doc_ids:
        active_passages = [p for p in evidence_passages if p.get("doc_id") not in excluded_doc_ids]

    report = []
    covered_count = 0
    partially_covered_count = 0
    not_covered_count = 0

    for item in syllabus_topics:
        topic_name = item.get("topic", "")
        subtopics = item.get("subtopics", [])

        # Retrieve evidence for the main topic
        main_retrieved = hybrid_search(topic_name, active_passages, top_k=4)

        # Check subtopic coverage
        subtopic_results = []
        for sub in subtopics:
            sub_res = hybrid_search(f"{topic_name} {sub}", active_passages, top_k=2)
            sub_score = sub_res[0]["score"] if sub_res else 0.0
            sub_covered = sub_score >= 0.25
            subtopic_results.append({
                "subtopic": sub,
                "is_covered": sub_covered,
                "score": sub_score,
                "evidence_passage": sub_res[0] if sub_covered and sub_res else None
            })

        covered_subs = sum(1 for s in subtopic_results if s["is_covered"])
        total_subs = max(1, len(subtopics))
        ratio = covered_subs / total_subs

        # Conservative categorization according to report Section 5.23
        if ratio >= 0.70 and main_retrieved and main_retrieved[0]["score"] >= 0.3:
            status = "Covered"
            covered_count += 1
            explanation = f"Comprehensive coverage found across {covered_subs}/{total_subs} subtopics with high textual support."
        elif ratio >= 0.30 or (main_retrieved and main_retrieved[0]["score"] >= 0.2):
            status = "Partially Covered"
            partially_covered_count += 1
            explanation = f"Partial coverage: {covered_subs}/{total_subs} subtopics mentioned. Certain aspects lack detailed explanations."
        else:
            status = "Not Covered"
            not_covered_count += 1
            explanation = "Adequate evidence was not found in the selected course materials for this topic."

        # Collect citations
        citations = []
        for p in main_retrieved[:2]:
            citations.append({
                "doc_name": p["doc_name"],
                "page": p["page"],
                "snippet": p["text"][:140] + "..."
            })

        report.append({
            "topic": topic_name,
            "status": status,
            "subtopics": subtopic_results,
            "coverage_percentage": int(ratio * 100),
            "citations": citations,
            "explanation": explanation
        })

    total_topics = len(syllabus_topics)
    overall_coverage = round((covered_count + (0.5 * partially_covered_count)) / max(1, total_topics) * 100, 1)

    return {
        "overall_coverage_percentage": overall_coverage,
        "total_topics": total_topics,
        "covered_count": covered_count,
        "partially_covered_count": partially_covered_count,
        "not_covered_count": not_covered_count,
        "topics_report": report
    }
