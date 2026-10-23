"""
Hybrid Lexical Retrieval: Combines Okapi BM25 and TF-IDF Ranking.
Provides passage deduplication, topic-sensitive filtering, and provenance tracking.
"""

import math
import re
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter


STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "cannot", "could", "did", "do", "does", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "her", "here",
    "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn't", "it",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not",
    "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out",
    "over", "own", "same", "shan't", "she", "should", "so", "some", "such", "than", "that", "the",
    "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "were", "weren't",
    "what", "when", "where", "which", "while", "who", "whom", "why", "with", "won't", "would", "you",
    "your", "yours", "yourself", "yourselves"
}


def tokenize(text: str) -> List[str]:
    """Extracts lowercased alphanumeric tokens, stripping punctuation."""
    tokens = re.findall(r"\b[a-zA-Z0-9_\-]{2,}\b", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


class BM25Retriever:
    """Okapi BM25 ranking algorithm with term-frequency saturation and length normalization."""
    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_lens = [len(tokenize(doc)) for doc in corpus]
        self.avgdl = (sum(self.doc_lens) / self.corpus_size) if self.corpus_size > 0 else 1.0

        # Build term frequencies per doc and document frequencies
        self.doc_freqs: Dict[str, int] = Counter()
        self.doc_term_freqs: List[Counter] = []

        for doc in corpus:
            tokens = tokenize(doc)
            tf = Counter(tokens)
            self.doc_term_freqs.append(tf)
            for token in tf.keys():
                self.doc_freqs[token] += 1

        # Precompute IDF
        self.idf: Dict[str, float] = {}
        for token, df in self.doc_freqs.items():
            # Standard Lucene / Robertson BM25 IDF formula
            self.idf[token] = math.log(1.0 + (self.corpus_size - df + 0.5) / (df + 0.5))

    def score(self, query_tokens: List[str]) -> List[float]:
        scores = [0.0] * self.corpus_size
        for token in query_tokens:
            if token not in self.idf:
                continue
            idf_val = self.idf[token]
            for doc_idx in range(self.corpus_size):
                tf = self.doc_term_freqs[doc_idx].get(token, 0)
                if tf > 0:
                    doc_len = self.doc_lens[doc_idx]
                    denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                    scores[doc_idx] += idf_val * (tf * (self.k1 + 1.0)) / denom
        return scores


class TFIDFRetriever:
    """TF-IDF vectorizer and cosine similarity ranking."""
    def __init__(self, corpus: List[str]):
        self.corpus_size = len(corpus)
        self.doc_term_freqs: List[Counter] = []
        self.doc_freqs: Dict[str, int] = Counter()

        for doc in corpus:
            tokens = tokenize(doc)
            tf = Counter(tokens)
            self.doc_term_freqs.append(tf)
            for token in tf.keys():
                self.doc_freqs[token] += 1

        self.idf: Dict[str, float] = {}
        for token, df in self.doc_freqs.items():
            self.idf[token] = math.log((1.0 + self.corpus_size) / (1.0 + df)) + 1.0

        # Precompute L2 norm per doc
        self.doc_norms = []
        for tf in self.doc_term_freqs:
            vec_sum_sq = sum((count * self.idf[t]) ** 2 for t, count in tf.items())
            self.doc_norms.append(math.sqrt(vec_sum_sq) if vec_sum_sq > 0 else 1.0)

    def score(self, query_tokens: List[str]) -> List[float]:
        scores = [0.0] * self.corpus_size
        q_tf = Counter(query_tokens)
        q_norm_sq = sum((count * self.idf.get(t, 1.0)) ** 2 for t, count in q_tf.items())
        q_norm = math.sqrt(q_norm_sq) if q_norm_sq > 0 else 1.0

        for token, count in q_tf.items():
            if token not in self.idf:
                continue
            token_idf = self.idf[token]
            q_weight = count * token_idf
            for doc_idx in range(self.corpus_size):
                doc_tf = self.doc_term_freqs[doc_idx].get(token, 0)
                if doc_tf > 0:
                    doc_weight = doc_tf * token_idf
                    scores[doc_idx] += q_weight * doc_weight

        # Normalize by cosine lengths
        for doc_idx in range(self.corpus_size):
            scores[doc_idx] /= (q_norm * self.doc_norms[doc_idx])
        return scores


def hybrid_search(
    query: str,
    passages: List[Dict[str, Any]],
    top_k: int = 5,
    min_score_threshold: float = 0.05,
    alpha_bm25: float = 0.6,
    alpha_tfidf: float = 0.4
) -> List[Dict[str, Any]]:
    """
    Combines BM25 and TF-IDF scores with length normalization and deduplication.
    Returns ranked passages with provenance metadata and scores.
    """
    if not passages or not query.strip():
        return []

    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    texts = [p["text"] for p in passages]

    # Calculate BM25
    bm25 = BM25Retriever(texts)
    bm25_raw = bm25.score(query_tokens)

    # Calculate TF-IDF
    tfidf = TFIDFRetriever(texts)
    tfidf_raw = tfidf.score(query_tokens)

    # Normalize scores to [0, 1]
    max_bm25 = max(bm25_raw) if bm25_raw and max(bm25_raw) > 0 else 1.0
    max_tfidf = max(tfidf_raw) if tfidf_raw and max(tfidf_raw) > 0 else 1.0

    scored_passages = []
    for idx, p in enumerate(passages):
        norm_bm25 = bm25_raw[idx] / max_bm25 if max_bm25 > 0 else 0.0
        norm_tfidf = tfidf_raw[idx] / max_tfidf if max_tfidf > 0 else 0.0
        combined = (alpha_bm25 * norm_bm25) + (alpha_tfidf * norm_tfidf)

        if combined >= min_score_threshold:
            entry = dict(p)
            entry["score"] = round(combined, 4)
            entry["bm25_score"] = round(norm_bm25, 4)
            entry["tfidf_score"] = round(norm_tfidf, 4)
            scored_passages.append(entry)

    # Sort descending by combined score
    scored_passages.sort(key=lambda x: x["score"], reverse=True)

    # Deduplicate overlapping passages
    deduped = _deduplicate_passages(scored_passages)

    return deduped[:top_k]


def _deduplicate_passages(passages: List[Dict[str, Any]], jaccard_thresh: float = 0.7) -> List[Dict[str, Any]]:
    """Removes near-duplicate passages that share more than jaccard_thresh word overlap."""
    unique: List[Dict[str, Any]] = []
    token_sets: List[set] = []

    for p in passages:
        p_tokens = set(tokenize(p["text"]))
        if not p_tokens:
            continue
        is_dup = False
        for seen in token_sets:
            inter = len(p_tokens.intersection(seen))
            union = len(p_tokens.union(seen))
            if union > 0 and (inter / union) >= jaccard_thresh:
                is_dup = True
                break
        if not is_dup:
            unique.append(p)
            token_sets.append(p_tokens)

    return unique
