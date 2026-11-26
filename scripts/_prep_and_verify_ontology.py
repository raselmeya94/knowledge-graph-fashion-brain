from knowledge_graph.config import load_settings
from knowledge_graph.pipeline.service import Platform
from knowledge_graph import ontology_registry as ormod

ormod._CACHE = None
s = load_settings()
p = Platform(s)
o = p.kg_registry.load_ontology_object("plmex")
print("loaded", o.name, o.version)
print("Buyer", "Buyer" in o.entity_types)
print("ProductionMethod", "ProductionMethod" in o.entity_types)

with p.graph._session() as session:
    row = session.run(
        r"MATCH (m:Material) WHERE m.name =~ '.*\d+.*' DETACH DELETE m RETURN count(*) AS c"
    ).single()
    print("deleted polluted materials", row["c"])
    for g in ["ecom_products", "plmex_products", "fashion_brain", "shared"]:
        n = 0
        while True:
            r = session.run(
                "MATCH (x:Entity {graph:$g}) WITH x LIMIT 200 DETACH DELETE x RETURN count(*) AS c",
                {"g": g},
            ).single()
            c = int(r["c"])
            n += c
            if c == 0:
                break
        print("wiped", g, n)

print("counts", p.graph.counts())
closer = getattr(p.graph, "close", None)
if callable(closer):
    closer()
