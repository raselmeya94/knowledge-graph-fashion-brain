"""Shared ontology: canonical entities, aliases, and the relationship allowlist."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from knowledge_graph.errors import RelationshipValidationError, ValidationFailed


class CanonicalEntity(BaseModel):
    id: str
    type: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    external_ids: list[str] = Field(default_factory=list)


class Ontology(BaseModel):
    name: str
    version: str
    description: str = ""
    entity_types: list[str]
    relationship_allowlist: dict[str, list[list[str]]]
    review_stopwords: list[str] = Field(default_factory=list)
    review_min_token_length: int = 5
    canonical_entities: list[CanonicalEntity]

    def entities_of_type(self, entity_type: str) -> list[CanonicalEntity]:
        return [entity for entity in self.canonical_entities if entity.type == entity_type]

    def get(self, canonical_id: str) -> CanonicalEntity | None:
        for entity in self.canonical_entities:
            if entity.id == canonical_id:
                return entity
        return None


STRUCTURAL_LABELS = {"Entity", "SourceEntity", "Canonical", "Record"}


def normalize_name(value: str) -> str:
    import re

    text = value.lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def empty_ontology(*, name: str = "empty", version: str = "0") -> Ontology:
    """Minimal ontology so the API can boot before any POST /ontologies."""

    return Ontology(
        name=name,
        version=version,
        description="Placeholder until ontologies are registered via API.",
        entity_types=["Entity"],
        relationship_allowlist={},
        canonical_entities=[],
    )


def load_ontology(path: Path) -> Ontology:
    data = json.loads(path.read_text(encoding="utf-8"))
    ontology = Ontology.model_validate(data)
    validate_ontology(ontology)
    return ontology


def validate_ontology(ontology: Ontology) -> None:
    known = set(ontology.entity_types)
    if len(known) != len(ontology.entity_types):
        raise ValidationFailed("Ontology entity types must be unique.")
    ids: set[str] = set()
    alias_index: dict[tuple[str, str], str] = {}
    external_index: dict[tuple[str, str], str] = {}
    for entity in ontology.canonical_entities:
        if entity.type not in known:
            raise ValidationFailed(f"Canonical entity {entity.id} uses unknown type {entity.type}.")
        if entity.id in ids:
            raise ValidationFailed(f"Duplicate canonical id {entity.id}.")
        ids.add(entity.id)
        names = [entity.name, *entity.aliases]
        for name in names:
            nn = normalize_name(name)
            # Symbol-only aliases (£, €, $) normalize to "" — skip, do not collide.
            if not nn:
                continue
            key = (entity.type, nn)
            owner = alias_index.get(key)
            if owner is not None and owner != entity.id:
                raise ValidationFailed(
                    f"Alias '{name}' for {entity.type} is claimed by both {owner} and {entity.id}."
                )
            alias_index[key] = entity.id
        for external_id in entity.external_ids:
            key = (entity.type, external_id)
            if key in external_index:
                raise ValidationFailed(f"External id {external_id} is duplicated for {entity.type}.")
            external_index[key] = entity.id
    for rel_type, pairs in ontology.relationship_allowlist.items():
        if not rel_type or rel_type != rel_type.upper():
            raise ValidationFailed(f"Relationship type {rel_type!r} must be upper snake case.")
        for pair in pairs:
            if len(pair) != 2 or pair[0] not in known or pair[1] not in known:
                raise ValidationFailed(f"Allowlist pair {pair} for {rel_type} is invalid.")


def domain_types(labels: list[str], ontology: Ontology) -> set[str]:
    known = set(ontology.entity_types)
    return {label for label in labels if label in known}


def validate_relationship(rel_type: str, start_labels: list[str], end_labels: list[str], ontology: Ontology) -> None:
    start_types = domain_types(start_labels, ontology)
    end_types = domain_types(end_labels, ontology)
    if rel_type == "RESOLVES_TO":
        if "SourceEntity" not in start_labels or "Canonical" not in end_labels:
            raise RelationshipValidationError(
                "RESOLVES_TO must link a SourceEntity to a Canonical entity."
            )
        if not (start_types & end_types):
            raise RelationshipValidationError(
                f"RESOLVES_TO type mismatch: {sorted(start_types)} -> {sorted(end_types)}."
            )
        return
    allowed = ontology.relationship_allowlist.get(rel_type)
    if allowed is None:
        raise RelationshipValidationError(f"Relationship type {rel_type} is not allowlisted.")
    for source_type, target_type in allowed:
        if source_type in start_types and target_type in end_types:
            return
    raise RelationshipValidationError(
        f"{rel_type} is not allowed from {sorted(start_types)} to {sorted(end_types)}."
    )


def aliases_for(entity: CanonicalEntity) -> list[str]:
    return [normalize_name(entity.name), *[normalize_name(alias) for alias in entity.aliases]]


def register_pair(ontology: Ontology, rel_type: str, start_type: str, end_type: str) -> None:
    """Add one relationship pair to an in-memory ontology."""

    for entity_type in (start_type, end_type):
        if entity_type not in ontology.entity_types:
            ontology.entity_types.append(entity_type)
    stored = ontology.relationship_allowlist.setdefault(rel_type, [])
    if [start_type, end_type] not in stored:
        stored.append([start_type, end_type])


def allow_relationship(ontology: Ontology, path: Path, rel_type: str, start_type: str, end_type: str) -> None:
    """Remember a relationship on the ontology file for this source. Other source files are not changed."""

    data = json.loads(path.read_text(encoding="utf-8"))
    types = data.setdefault("entity_types", [])
    allowlist = data.setdefault("relationship_allowlist", {})
    changed = False
    for entity_type in (start_type, end_type):
        if entity_type not in types:
            types.append(entity_type)
            changed = True
    pairs = allowlist.setdefault(rel_type, [])
    if [start_type, end_type] not in pairs:
        pairs.append([start_type, end_type])
        changed = True
    register_pair(ontology, rel_type, start_type, end_type)
    if changed:
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def combine_ontologies(product: Ontology, news: Ontology) -> Ontology:
    """Validation view. The product and news files stay separate."""

    entity_types = list(product.entity_types)
    allowlist = {key: [list(pair) for pair in pairs] for key, pairs in product.relationship_allowlist.items()}
    canonical_entities = list(product.canonical_entities)
    canonical_ids = {entity.id for entity in canonical_entities}
    for entity_type in news.entity_types:
        if entity_type not in entity_types:
            entity_types.append(entity_type)
    for rel_type, pairs in news.relationship_allowlist.items():
        bucket = allowlist.setdefault(rel_type, [])
        for pair in pairs:
            if pair not in bucket:
                bucket.append(list(pair))
    for entity in news.canonical_entities:
        if entity.id not in canonical_ids:
            canonical_entities.append(entity)
            canonical_ids.add(entity.id)
    return Ontology(
        name=product.name,
        version=product.version,
        description=product.description,
        entity_types=entity_types,
        relationship_allowlist=allowlist,
        review_stopwords=list(product.review_stopwords),
        review_min_token_length=product.review_min_token_length,
        canonical_entities=canonical_entities,
    )
