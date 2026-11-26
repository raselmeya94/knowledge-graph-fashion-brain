// Fashion Brain connectivity checks — run in Neo4j Browser after full ecom + plmex rebuild.
// Expect: polluted materials = 0, inventory collisions = 0, shared hubs > 0 for Brand/Material/Color/TargetGroup/Reference.

// 1) Product counts by graph
MATCH (p:Product)
WHERE coalesce(p.lifecycle,'active') <> 'deleted'
RETURN p.graph AS graph, coalesce(p.source, p.prov_source_type) AS source, count(p) AS n
ORDER BY n DESC;

// 2) Polluted Material nodes (MUST be 0)
MATCH (m:Material)
WHERE m.name =~ '.*\d+.*'
RETURN count(m) AS polluted_count, collect(m.name)[0..20] AS samples;

// 3) InventoryRecord collisions (MUST be 0)
MATCH (p:Product)-[:HAS_INVENTORY_RECORD]->(inv:InventoryRecord)
WITH inv, count(p) AS prod_count
WHERE prod_count > 1
RETURN count(inv) AS collided_inventory_nodes;

// 4) Shared hubs between plmex and ecom
MATCH (plm:Product)-->(hub)<--(ecom:Product)
WHERE (plm.source STARTS WITH 'plmex' OR plm.graph = 'plmex_products')
  AND (ecom.source = 'ecom' OR ecom.graph = 'ecom_products')
  AND plm.id <> ecom.id
RETURN coalesce([x IN labels(hub) WHERE NOT x IN ['Entity','SourceEntity','Record','Canonical']][0], labels(hub)[0]) AS shared_hub_type,
       count(DISTINCT hub) AS shared_nodes
ORDER BY shared_nodes DESC;

// 5) Shared Brand detail
MATCH (plm:Product)-[:HAS_BRAND]->(b:Brand)<-[:HAS_BRAND]-(ecom:Product)
WHERE (plm.source STARTS WITH 'plmex' OR plm.graph = 'plmex_products')
  AND (ecom.source = 'ecom' OR ecom.graph = 'ecom_products')
RETURN b.id AS brand_id, b.name AS brand,
       count(DISTINCT plm) AS plm_products,
       count(DISTINCT ecom) AS ecom_products
ORDER BY ecom_products + plm_products DESC
LIMIT 20;

// 6) Shared Material detail (clean fibers only)
MATCH (plm:Product)-[:USES_MATERIAL|PRIMARY_MATERIAL]->(m:Material)<-[:USES_MATERIAL|PRIMARY_MATERIAL]-(ecom:Product)
WHERE (plm.source STARTS WITH 'plmex' OR plm.graph = 'plmex_products')
  AND (ecom.source = 'ecom' OR ecom.graph = 'ecom_products')
RETURN m.id AS material_id, m.name AS material,
       count(DISTINCT plm) AS plm_products,
       count(DISTINCT ecom) AS ecom_products
ORDER BY ecom_products + plm_products DESC
LIMIT 20;

// 7) Shared Reference + RESOLVES_TO
MATCH (plm:Product)-[:HAS_REFERENCE]->(r:Reference)<-[:HAS_REFERENCE]-(ecom:Product)
WHERE (plm.source STARTS WITH 'plmex' OR plm.graph = 'plmex_products')
  AND (ecom.source = 'ecom' OR ecom.graph = 'ecom_products')
  AND plm.id <> ecom.id
OPTIONAL MATCH (plm)-[rel:RESOLVES_TO]->(ecom)
RETURN count(DISTINCT r) AS shared_refs,
       count(DISTINCT plm) AS plm_with_shared_ref,
       count(DISTINCT ecom) AS ecom_with_shared_ref,
       count(rel) AS resolves_to_edges;

// 8) TargetGroup bridge
MATCH (plm:Product)-[:FOR_TARGET_GROUP]->(tg:TargetGroup)<-[:FOR_TARGET_GROUP]-(ecom:Product)
WHERE (plm.source STARTS WITH 'plmex' OR plm.graph = 'plmex_products')
  AND (ecom.source = 'ecom' OR ecom.graph = 'ecom_products')
RETURN tg.id AS id, tg.name AS name, count(DISTINCT plm) AS plm_n, count(DISTINCT ecom) AS ecom_n
ORDER BY plm_n + ecom_n DESC;

// 9) fashion_brain hub inventory
MATCH (n:Entity {graph:'fashion_brain'})
WHERE coalesce(n.lifecycle,'active') <> 'deleted'
RETURN coalesce([x IN labels(n) WHERE NOT x IN ['Entity','SourceEntity','Record','Canonical']][0], head(labels(n))) AS kind,
       count(*) AS n
ORDER BY n DESC;

// 10) Orphan products (no brand AND no material)
MATCH (p:Product)
WHERE coalesce(p.lifecycle,'active') <> 'deleted'
  AND NOT (p)-[:HAS_BRAND]->()
  AND NOT (p)-[:USES_MATERIAL|PRIMARY_MATERIAL]->()
RETURN coalesce(p.graph, p.source) AS graph, count(p) AS orphan_count;
