import pytest
from src.study_tools import generate_flashcards, generate_quiz
from src.library import DocumentLibrary
from src.citation_validator import validate_citations, check_answer_grounding
from src.source_analysis import evaluate_syllabus_coverage, DEFAULT_OS_SYLLABUS, render_mind_map_svg

@pytest.fixture
def sample_passages():
    lib = DocumentLibrary()
    return lib.get_passages_for_docs()

def test_flashcard_generation(sample_passages):
    deck_res = generate_flashcards("Deadlocks and Coffman conditions", sample_passages, requested_count=3)
    assert deck_res["success"] is True
    assert len(deck_res["cards"]) == 3
    for card in deck_res["cards"]:
        assert "front" in card and len(card["front"]) > 5
        assert "back" in card and len(card["back"]) > 5
        assert "id" in card
        assert card["id"].startswith("fc_")

def test_quiz_generation(sample_passages):
    quiz_res = generate_quiz("Page Replacement FIFO LRU", sample_passages, num_questions=3)
    assert quiz_res["success"] is True
    assert len(quiz_res["questions"]) == 3
    for q in quiz_res["questions"]:
        assert len(q["options"]) == 4
        assert 0 <= q["correct_index"] < 4
        assert q["options"][q["correct_index"]] == q["correct_answer"]
        assert "explanation" in q

def test_citation_and_grounding_validation(sample_passages):
    # Passages on CPU scheduling
    retrieved = [p for p in sample_passages if "cpu" in p["doc_name"].lower()][:2]
    c_res = validate_citations(retrieved, retrieved)
    assert c_res["citation_count"] == len(retrieved)
    assert c_res["all_verified"] is True

    answer = "CPU scheduling allows switching between processes to optimize CPU utilization and throughput."
    g_res = check_answer_grounding(answer, retrieved)
    assert g_res["is_grounded"] is True
    assert g_res["grounding_score"] > 0.5

def test_syllabus_coverage_evaluation(sample_passages):
    cov = evaluate_syllabus_coverage(DEFAULT_OS_SYLLABUS, sample_passages)
    assert cov["total_topics"] == len(DEFAULT_OS_SYLLABUS)
    assert cov["overall_coverage_percentage"] > 50.0
    assert cov["covered_count"] >= 3

def test_mind_map_svg_rendering():
    dummy_graph = {
        "root_node": "Deadlocks",
        "nodes": [
            {"id": "node_root", "label": "Deadlocks", "category": "root", "description": "Deadlocks overview"},
            {"id": "n1", "label": "Mutual Exclusion", "category": "condition", "description": "Non-shareable resource"},
            {"id": "n2", "label": "Circular Wait", "category": "condition", "description": "Closed chain of waiting"},
        ],
        "edges": [
            {"from": "node_root", "to": "n1", "label": "requires", "status": "source_supported", "citation": "Page 1"},
            {"from": "node_root", "to": "n2", "label": "requires", "status": "inferred", "citation": "Inferred"}
        ]
    }
    svg = render_mind_map_svg(dummy_graph)
    assert "<svg" in svg
    assert "</svg>" in svg
    assert "Source-Supported" in svg
    assert "Inferred" in svg
    assert "Deadlocks" in svg

def test_custom_syllabus_parsing():
    from src.source_analysis import parse_custom_syllabus_text
    sample_text = (
        "CPU Scheduling: FCFS, SJF, Round Robin\n"
        "Deadlocks: Coffman Conditions, Banker's Algorithm\n"
        "Memory Management: Paging, Page Faults"
    )
    parsed = parse_custom_syllabus_text(sample_text)
    assert len(parsed) == 3
    assert parsed[0]["topic"] == "CPU Scheduling"
    assert "FCFS" in parsed[0]["subtopics"]
    assert "SJF" in parsed[0]["subtopics"]
    assert parsed[1]["topic"] == "Deadlocks"
    assert "Banker's Algorithm" in parsed[1]["subtopics"]

