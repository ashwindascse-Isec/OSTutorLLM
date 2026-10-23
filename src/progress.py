"""
Progress Tracking and Revision Recommendation Engine.
Tracks flashcard mastery, quiz performance per topic,
identifies learning gaps, and generates explainable revision recommendations.
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.config import PROGRESS_FILE


class ProgressTracker:
    def __init__(self, filepath: Optional[Path] = None):
        self.filepath = filepath or PROGRESS_FILE
        self._ensure_file()

    def _ensure_file(self):
        if not self.filepath.exists():
            default_data = {
                "flashcards": {},
                "quizzes": {},
                "topic_stats": {},
                "activity_log": []
            }
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(default_data, f, indent=2)

    def load_data(self) -> Dict[str, Any]:
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"flashcards": {}, "quizzes": {}, "topic_stats": {}, "activity_log": []}

    def save_data(self, data: Dict[str, Any]):
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def record_flashcard_review(self, card_id: str, front: str, topic: str, is_mastered: bool):
        data = self.load_data()
        cards = data.setdefault("flashcards", {})
        topic_stats = data.setdefault("topic_stats", {})

        entry = cards.get(card_id, {
            "id": card_id,
            "front": front,
            "topic": topic,
            "reviews": 0,
            "mastered_count": 0,
            "missed_count": 0,
            "last_reviewed": ""
        })

        entry["reviews"] += 1
        if is_mastered:
            entry["mastered_count"] += 1
        else:
            entry["missed_count"] += 1
        entry["last_reviewed"] = time.strftime("%Y-%m-%d %H:%M:%S")
        cards[card_id] = entry

        # Update topic level stats
        t_stat = topic_stats.setdefault(topic, {"reviews": 0, "correct": 0, "total": 0})
        t_stat["reviews"] += 1
        t_stat["total"] += 1
        if is_mastered:
            t_stat["correct"] += 1

        data["activity_log"].append({
            "type": "flashcard",
            "topic": topic,
            "result": "mastered" if is_mastered else "missed",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })

        self.save_data(data)

    def record_quiz_attempt(
        self,
        question_id: str,
        question: str,
        topic: str,
        selected_option: str,
        correct_answer: str,
        is_correct: bool
    ):
        data = self.load_data()
        quizzes = data.setdefault("quizzes", {})
        topic_stats = data.setdefault("topic_stats", {})

        entry = quizzes.get(question_id, {
            "id": question_id,
            "question": question,
            "topic": topic,
            "attempts": 0,
            "correct_count": 0,
            "incorrect_count": 0,
            "last_attempt": ""
        })

        entry["attempts"] += 1
        if is_correct:
            entry["correct_count"] += 1
        else:
            entry["incorrect_count"] += 1
        entry["last_attempt"] = time.strftime("%Y-%m-%d %H:%M:%S")
        quizzes[question_id] = entry

        # Topic stats
        t_stat = topic_stats.setdefault(topic, {"reviews": 0, "correct": 0, "total": 0})
        t_stat["reviews"] += 1
        t_stat["total"] += 1
        if is_correct:
            t_stat["correct"] += 1

        data["activity_log"].append({
            "type": "quiz",
            "topic": topic,
            "result": "correct" if is_correct else "incorrect",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })

        self.save_data(data)

    def get_progress_summary(self) -> Dict[str, Any]:
        data = self.load_data()
        cards = data.get("flashcards", {})
        quizzes = data.get("quizzes", {})
        topic_stats = data.get("topic_stats", {})

        total_cards_reviewed = sum(c.get("reviews", 0) for c in cards.values())
        mastered_cards = sum(1 for c in cards.values() if c.get("mastered_count", 0) > c.get("missed_count", 0))
        total_unique_cards = len(cards)

        total_quiz_attempts = sum(q.get("attempts", 0) for q in quizzes.values())
        correct_quizzes = sum(q.get("correct_count", 0) for q in quizzes.values())
        quiz_accuracy = round((correct_quizzes / total_quiz_attempts * 100), 1) if total_quiz_attempts > 0 else 0.0

        topic_breakdown = {}
        for t, stats in topic_stats.items():
            tot = stats.get("total", 0)
            corr = stats.get("correct", 0)
            acc = round((corr / tot * 100), 1) if tot > 0 else 0.0
            topic_breakdown[t] = {
                "total_attempts": tot,
                "correct": corr,
                "accuracy": acc,
                "status": "Needs Review" if acc < 60 else ("Proficient" if acc >= 80 else "Developing")
            }

        return {
            "total_cards_reviewed": total_cards_reviewed,
            "unique_cards_practiced": total_unique_cards,
            "mastered_cards_count": mastered_cards,
            "card_mastery_rate": round((mastered_cards / total_unique_cards * 100), 1) if total_unique_cards > 0 else 0.0,
            "total_quiz_questions": len(quizzes),
            "total_quiz_attempts": total_quiz_attempts,
            "quiz_accuracy_rate": quiz_accuracy,
            "topic_breakdown": topic_breakdown
        }

    def get_targeted_revision_recommendations(self) -> List[Dict[str, Any]]:
        """
        Generates explainable revision recommendations prioritizing items
        with high miss rates or topics with accuracy < 60%.
        """
        data = self.load_data()
        recommendations = []

        # 1. Inspect missed flashcards
        for c in data.get("flashcards", {}).values():
            misses = c.get("missed_count", 0)
            mastered = c.get("mastered_count", 0)
            if misses > mastered:
                recommendations.append({
                    "type": "Flashcard",
                    "item_id": c["id"],
                    "title": c["front"],
                    "topic": c.get("topic", "Operating Systems"),
                    "priority": "High" if misses >= 2 else "Medium",
                    "reason": f"Marked as missed {misses} time(s). Practice active recall to solidify this concept."
                })

        # 2. Inspect failed quiz questions
        for q in data.get("quizzes", {}).values():
            in_corr = q.get("incorrect_count", 0)
            corr = q.get("correct_count", 0)
            if in_corr > corr:
                recommendations.append({
                    "type": "Quiz Question",
                    "item_id": q["id"],
                    "title": q["question"],
                    "topic": q.get("topic", "Operating Systems"),
                    "priority": "High",
                    "reason": f"Answered incorrectly {in_corr} time(s). Review relevant textbook chapter and re-test."
                })

        # Sort recommendations by priority
        priority_order = {"High": 0, "Medium": 1, "Low": 2}
        recommendations.sort(key=lambda x: priority_order.get(x["priority"], 3))
        return recommendations

    def reset_progress(self):
        default_data = {
            "flashcards": {},
            "quizzes": {},
            "topic_stats": {},
            "activity_log": []
        }
        self.save_data(default_data)
