"""Deterministic entity resolution. Similar names become reviews, not merges."""

from __future__ import annotations

from knowledge_graph.models import MatchDecision, ReviewDecision
from knowledge_graph.ontology import CanonicalEntity, Ontology, aliases_for, normalize_name


class TokenJaccardScorer:
    """Lexical similarity used only to propose review candidates."""

    def __init__(self, stopwords: set[str], min_length: int):
        self.stopwords = stopwords
        self.min_length = min_length

    def _tokens(self, value: str) -> set[str]:
        return {
            token
            for token in normalize_name(value).split()
            if len(token) >= self.min_length and token not in self.stopwords
        }

    def score(self, left: str, right: str) -> float:
        left_tokens = self._tokens(left)
        right_tokens = self._tokens(right)
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def informative_tokens(value: str, ontology: Ontology) -> set[str]:
    stop = set(ontology.review_stopwords)
    minimum = ontology.review_min_token_length
    return {token for token in normalize_name(value).split() if len(token) >= minimum and token not in stop}


def _canonical_tokens(entity: CanonicalEntity, ontology: Ontology) -> set[str]:
    tokens: set[str] = set()
    for name in [entity.name, *entity.aliases]:
        tokens |= informative_tokens(name, ontology)
    return tokens


def match_entity(
    *,
    name: str,
    entity_type: str,
    external_id: str | None,
    ontology: Ontology,
    decisions: list[ReviewDecision],
    scorer=None,
    review_threshold: float = 0.5,
) -> list[MatchDecision]:
    human = [
        decision
        for decision in decisions
        if decision.start_type == entity_type and normalize_name(decision.start_name) == normalize_name(name)
    ]
    rejected = {decision.canonical_id for decision in human if decision.status == "rejected"}
    approved = [decision for decision in human if decision.status == "approved"]

    def verified(method: str, entity: CanonicalEntity, evidence: str) -> MatchDecision:
        return MatchDecision(
            status="verified",
            method=method,
            canonical_id=entity.id,
            similarity=None,
            evidence=evidence,
            entity_type=entity_type,
            source_name=name,
        )

    def rejected_decision(entity: CanonicalEntity, evidence: str) -> MatchDecision:
        return MatchDecision(
            status="rejected",
            method="human_review",
            canonical_id=entity.id,
            similarity=None,
            evidence=evidence,
            entity_type=entity_type,
            source_name=name,
        )

    if external_id:
        hits = [entity for entity in ontology.entities_of_type(entity_type) if external_id in entity.external_ids]
        if len(hits) == 1:
            entity = hits[0]
            if entity.id in rejected:
                return [rejected_decision(entity, f"Stable id {external_id} was rejected by a recorded review decision.")]
            return [
                verified(
                    "exact_stable_id",
                    entity,
                    f"External id {external_id} matches canonical {entity.id}.",
                )
            ]
        if len(hits) > 1:
            return [
                MatchDecision(
                    status="review",
                    method="ambiguous_stable_id",
                    canonical_id=entity.id,
                    similarity=None,
                    evidence=f"External id {external_id} matches more than one canonical entity.",
                    entity_type=entity_type,
                    source_name=name,
                )
                for entity in hits
                if entity.id not in rejected
            ]

    alias_hits = [
        entity
        for entity in ontology.entities_of_type(entity_type)
        if normalize_name(name) in set(aliases_for(entity))
    ]
    if len(alias_hits) == 1:
        entity = alias_hits[0]
        if entity.id in rejected:
            return [rejected_decision(entity, f"Curated alias for {name!r} was rejected by a recorded review decision.")]
        method = "exact_name" if normalize_name(name) == normalize_name(entity.name) else "curated_alias"
        return [verified(method, entity, f"{name!r} is a curated name for {entity.id}.")]

    if approved:
        results = []
        for decision in approved:
            entity = ontology.get(decision.canonical_id)
            if entity and entity.type == entity_type:
                results.append(
                    verified(
                        "human_review",
                        entity,
                        decision.note or f"Approved by a recorded review decision for {entity.id}.",
                    )
                )
        if results:
            return results

    source_tokens = informative_tokens(name, ontology)
    reviews: list[MatchDecision] = []
    for entity in ontology.entities_of_type(entity_type):
        shared = source_tokens & _canonical_tokens(entity, ontology)
        if not shared:
            continue
        union = source_tokens | _canonical_tokens(entity, ontology)
        similarity = len(shared) / len(union) if union else 0.0
        if entity.id in rejected:
            reviews.append(
                rejected_decision(
                    entity,
                    f"Shared token(s) {sorted(shared)} were rejected for {entity.id}.",
                )
            )
            continue
        reviews.append(
            MatchDecision(
                status="review",
                method="shared_token",
                canonical_id=entity.id,
                similarity=similarity,
                evidence=(
                    f"Shared informative token(s) {sorted(shared)} between {name!r} and {entity.name!r}. "
                    "This is not a verified merge."
                ),
                entity_type=entity_type,
                source_name=name,
            )
        )

    if scorer is not None:
        already = {item.canonical_id for item in reviews}
        for entity in ontology.entities_of_type(entity_type):
            if entity.id in already or entity.id in rejected:
                continue
            score = float(scorer.score(name, entity.name))
            if score >= review_threshold:
                reviews.append(
                    MatchDecision(
                        status="review",
                        method="similarity_review",
                        canonical_id=entity.id,
                        similarity=score,
                        evidence=(
                            f"Similarity score {score:.2f} against {entity.name!r} is only a review candidate."
                        ),
                        entity_type=entity_type,
                        source_name=name,
                    )
                )

    if not reviews:
        return [
            MatchDecision(
                status="unresolved",
                method=None,
                canonical_id=None,
                similarity=None,
                evidence="No exact identifier, curated alias, or review candidate.",
                entity_type=entity_type,
                source_name=name,
            )
        ]
    return reviews
