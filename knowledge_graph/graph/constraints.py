"""Neo4j constraints and indexes. Memory repositories record the same statements."""

CONSTRAINTS = [
    "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE",
]

INDEXES = [
    "CREATE INDEX entity_graph IF NOT EXISTS FOR (n:Entity) ON (n.graph)",
    "CREATE INDEX entity_source IF NOT EXISTS FOR (n:Entity) ON (n.source_id)",
    "CREATE INDEX entity_name IF NOT EXISTS FOR (n:Entity) ON (n.name)",
    "CREATE INDEX entity_lifecycle IF NOT EXISTS FOR (n:Entity) ON (n.lifecycle)",
]

SCHEMA_STATEMENTS = CONSTRAINTS + INDEXES
