import os
import re
from pathlib import Path

import psycopg

for line in Path(r"d:\Rasel-P\R&D-KG\.env").read_text(encoding="utf-8").splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k in ("FASHIONGPT_ECOM_DSN", "FASHIONGPT_PLMEX_DSN"):
            os.environ[k] = v


def norms(dsn: str, table: str, col: str) -> set[str]:
    out: set[str] = set()
    with psycopg.connect(dsn) as c, c.cursor() as cur:
        cur.execute(
            f"""
            SELECT DISTINCT btrim(CAST({col} AS text))
            FROM {table}
            WHERE {col} IS NOT NULL AND btrim(CAST({col} AS text)) <> ''
            """
        )
        for (v,) in cur.fetchall():
            t = re.sub(r"[^a-z0-9]+", " ", v.strip().lower())
            t = re.sub(r"\s+", " ", t).strip()
            if t and len(t) >= 4:
                out.add(t)
    return out


e = os.environ["FASHIONGPT_ECOM_DSN"]
p = os.environ["FASHIONGPT_PLMEX_DSN"]
er = norms(e, "master_products", "reference") | norms(e, "master_products", "display_reference")
pr = norms(p, "plmex_products", "style_reference") | norms(
    p, "plmex_products", "customer_style_reference"
)
inter = er & pr
print("refs len>=4 shared", len(inter))
print("examples", sorted(inter, key=len)[:20])
print("ecom_n", len(er), "plm_n", len(pr))
