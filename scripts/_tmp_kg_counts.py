from knowledge_graph.config import load_settings
from knowledge_graph.pipeline.service import Platform

p = Platform(load_settings())
print("counts", p.graph.counts())
with p.graph._session() as s:
    for lab, q in [
        ("ecom", "MATCH (n:Product {graph:'ecom_products'}) RETURN count(n) AS n"),
        ("plmex", "MATCH (n:Product {graph:'plmex_products'}) RETURN count(n) AS n"),
        ("hubs", "MATCH (n:Entity {graph:'fashion_brain'}) RETURN count(n) AS n"),
    ]:
        print(lab, s.run(q).single()["n"])
