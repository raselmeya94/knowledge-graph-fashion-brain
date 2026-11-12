"""Fashion Brain linking helpers (question → candidates). KG owns Neo4j lookup APIs."""

from fashion_brain.linking.question_candidates import candidates_payload, link_question_candidates

__all__ = ["link_question_candidates", "candidates_payload"]
