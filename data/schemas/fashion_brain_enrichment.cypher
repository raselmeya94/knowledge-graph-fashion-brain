// Fashion Brain post-sync enrichment
// Generated from semantic_bridges.json — prefer run_enrichment(platform) for parameterized execution.

// params: {"rows": [{"group": "Inditex", "buyers": ["INDITEX", "INDITEX(ZARA KIDS)", "INDITEX(ZARA LADY)", "INDITEX(PULL&BEAR)", "INDITEX(STRADIVARIUS)"], "brands": ["zara", "zara kids", "zara lady", "zara home
UNWIND $rows AS row
                MERGE (g:Entity:RetailGroup {id: 'group:' + toLower(replace(replace(row.group, '&', 'and'), ' ', '-'))})
                SET g.name = row.group, g.normalized_name = toLower(row.group), g.graph = 'fashion_brain', g.lifecycle = 'active'
                WITH g, row
                UNWIND row.buyers AS bName
                OPTIONAL MATCH (by:Buyer)
                WHERE toUpper(coalesce(by.name, '')) = toUpper(bName)
                FOREACH (_ IN CASE WHEN by IS NOT NULL THEN [1] ELSE [] END |
                  MERGE (by)-[:IN_GROUP]->(g)
                )
                WITH g, row
                UNWIND row.brands AS brName
                OPTIONAL MATCH (br:Brand)
                WHERE toLower(coalesce(br.name, '')) = toLower(brName)
                   OR toLower(coalesce(br.normalized_name, '')) = toLower(brName)
                FOREACH (_ IN CASE WHEN br IS NOT NULL THEN [1] ELSE [] END |
                  MERGE (br)-[:IN_GROUP]->(g)
                );

// params: {"rows": [{"mat": "cotton", "grp": "natural"}, {"mat": "wool", "grp": "natural"}, {"mat": "linen", "grp": "natural"}, {"mat": "silk", "grp": "natural"}, {"mat": "polyester", "grp": "synthetic"}, {"mat
UNWIND $rows AS m
                MATCH (mat:Material)
                WHERE toLower(coalesce(mat.name, '')) = toLower(m.mat)
                   OR toLower(coalesce(mat.normalized_name, '')) = toLower(m.mat)
                MERGE (mg:Entity:MaterialGroup {id: 'materialgroup:' + m.grp})
                SET mg.name = m.grp, mg.normalized_name = m.grp, mg.graph = 'fashion_brain', mg.lifecycle = 'active'
                MERGE (mat)-[:IN_MATERIAL_GROUP]->(mg);

// params: {"rows": [{"colors": ["white", "off white", "off-white", "beige", "khaki", "cream", "ecru"], "fam": "neutrals"}, {"colors": ["black", "grey", "dark grey", "light grey", "white and black"], "fam": "mon
UNWIND $rows AS cf
                MERGE (f:Entity:ColorFamily {id: 'colorfamily:' + cf.fam})
                SET f.name = cf.fam, f.normalized_name = cf.fam, f.graph = 'fashion_brain', f.lifecycle = 'active'
                WITH f, cf
                UNWIND cf.colors AS cName
                MATCH (col:Color)
                WHERE toLower(coalesce(col.name, '')) = toLower(cName)
                   OR toLower(coalesce(col.normalized_name, '')) = toLower(cName)
                MERGE (col)-[:IN_COLOR_FAMILY]->(f);

// params: {}
MATCH (plm:Product)-[:HAS_REFERENCE]->(r:Reference)<-[:HAS_REFERENCE]-(ecom:Product)
            WHERE (plm.source STARTS WITH 'plmex' OR plm.graph = 'plmex_products')
              AND (ecom.source = 'ecom' OR ecom.graph = 'ecom_products')
              AND plm.id <> ecom.id
            MERGE (plm)-[rel:RESOLVES_TO]->(ecom)
            SET rel.via = 'reference', rel.ref = coalesce(r.name, r.id), rel.confidence = 0.95, rel.lifecycle = 'active';

// params: {}
MATCH (t:AiMaterialTag), (m:Material)
            WHERE toLower(coalesce(t.name, '')) CONTAINS toLower(coalesce(m.name, ''))
               OR toLower(coalesce(t.normalized_name, '')) CONTAINS toLower(coalesce(m.normalized_name, ''))
            MERGE (t)-[:REFERS_TO_MATERIAL]->(m);

// params: {"rows": [{"id": "cat:cardigan", "name": "cardigan", "parent": "knitwear", "aliases": ["cardigans", "cardigan", "knitted cardigans", "Cardigan Sweater", "Knit Cardigan", "Woven Cardigan", "Top Cardiga
UNWIND $rows AS row
                MERGE (c:Entity:CanonicalCategory {id: row.id})
                SET c.name = row.name, c.normalized_name = toLower(row.name),
                    c.parent = row.parent, c.graph = 'fashion_brain', c.lifecycle = 'active'
                WITH c, row
                UNWIND row.aliases AS alias
                OPTIONAL MATCH (n)
                WHERE (n:Subfamily OR n:GarmentType OR n:PlmSubSubCategory OR n:PlmSubCategory OR n:GarmentCategory OR n:Family)
                  AND (
                    toLower(coalesce(n.name, '')) = toLower(alias)
                    OR toLower(coalesce(n.normalized_name, '')) = toLower(alias)
                  )
                FOREACH (_ IN CASE WHEN n IS NOT NULL THEN [1] ELSE [] END |
                  MERGE (n)-[:MAPS_TO_CATEGORY]->(c)
                );

