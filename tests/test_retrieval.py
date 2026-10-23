import pytest
from src.library import DocumentLibrary
from src.retrieval import hybrid_search, tokenize, BM25Retriever, TFIDFRetriever

@pytest.fixture
def library():
    return DocumentLibrary()

def test_library_seeding_and_retrieval(library):
    docs = library.list_documents()
    assert len(docs) >= 5, "Expected default OS chapters to be seeded"

    passages = library.get_passages_for_docs()
    assert len(passages) > 10, "Expected passages to be chunked and indexed"

    # Search for deadlock conditions
    results = hybrid_search("four necessary conditions for deadlock", passages, top_k=3)
    assert len(results) > 0
    top = results[0]
    assert "deadlock" in top["doc_name"].lower() or "deadlock" in top["text"].lower()
    assert any(term in top["text"].lower() for term in ["mutual exclusion", "circular wait", "hold and wait"])

def test_page_replacement_query(library):
    passages = library.get_passages_for_docs()
    results = hybrid_search("page fault handling demand paging", passages, top_k=3)
    assert len(results) > 0
    top = results[0]
    assert "page fault" in top["text"].lower()

def test_deduplication():
    passages = [
        {"id": "p1", "text": "CPU scheduling is the basis of multiprogrammed operating systems.", "doc_name": "OS", "page": 1, "chunk_idx": 1},
        {"id": "p2", "text": "CPU scheduling is the basis of multiprogrammed operating systems and computers.", "doc_name": "OS", "page": 1, "chunk_idx": 2},
        {"id": "p3", "text": "Deadlock circular wait condition mutual exclusion.", "doc_name": "OS", "page": 2, "chunk_idx": 3},
    ]
    res = hybrid_search("CPU scheduling multiprogrammed", passages, top_k=5)
    # The two near-duplicate passages should be deduplicated
    assert len(res) == 1
