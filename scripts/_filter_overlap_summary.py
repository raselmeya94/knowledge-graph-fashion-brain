import json
import re
from pathlib import Path

p = Path(r"d:\Rasel-P\R&D-KG\data\generated\eval\RAW_VALUE_OVERLAP_ECOM_PLMEX.json")
d = json.loads(p.read_text(encoding="utf-8"))

numish = set(
    """in_stock_count out_stock_count image_count color_count colorway_tile_count
       knitting_time weight gsm current_stock shipment_quantity gauge cuttable_width
       season_year unit""".split()
)


def mostly_numeric(samples: list[str]) -> bool:
    if not samples:
        return False
    n = sum(1 for s in samples if re.fullmatch(r"\d+(\.\d+)?", s))
    return n / len(samples) >= 0.6


print("MEANINGFUL pairs (non-numeric tokens):")
meaningful = []
for r in d["top_pairs"]:
    if r["ecom"] in numish or r["plmex"] in numish:
        continue
    if mostly_numeric(r["sample_shared"]):
        continue
    if r["shared"] < 3:
        continue
    meaningful.append(r)
    print(
        f"{r['ecom']:24} <-> {r['plmex']:24} "
        f"shared={r['shared']:4d} jac={r['jaccard']:.3f} "
        f"covE={r['cov_ecom']:.3f} covP={r['cov_plmex']:.3f}  "
        f"{r['sample_shared'][:6]}"
    )

print("\nSame / near-same column names:")
for r in d["top_pairs"]:
    a, b = r["ecom"], r["plmex"]
    if a == b or a.rstrip("s") == b.rstrip("s") or a + "s" == b or b + "s" == a:
        print(r["ecom"], "<->", r["plmex"], "shared", r["shared"], r["sample_shared"][:8])

print("\nAll brand pairs in top_pairs:")
for r in d["top_pairs"]:
    if r["ecom"] == "brand" or r["plmex"] in {"brand", "buyer"}:
        if r["ecom"] == "brand" or r["plmex"] in {"brand", "buyer"}:
            if r["ecom"] == "brand":
                print(r)

# rewrite focused md
lines = [
    "# Raw value overlap (Postgres only)",
    "",
    "Compared distinct normalized tokens in `master_products` vs `plmex_products`.",
    "No Neo4j / relationships. Numeric-noise pairs (stock counts vs knitting_time, etc.) dropped.",
    "",
    "## Real content matches",
    "",
    "| Ecom column | PLMEX column | Shared values | Jaccard | Cover ecom | Cover plmex | Example shared values |",
    "| --- | --- | ---: | ---: | ---: | ---: | --- |",
]
# dedupe keep highest shared per pair already sorted
seen = set()
for r in meaningful:
    key = (r["ecom"], r["plmex"])
    if key in seen:
        continue
    seen.add(key)
    samples = ", ".join(r["sample_shared"][:8])
    lines.append(
        f"| `{r['ecom']}` | `{r['plmex']}` | {r['shared']} | {r['jaccard']:.3f} | "
        f"{r['cov_ecom']:.3f} | {r['cov_plmex']:.3f} | {samples} |"
    )

lines += [
    "",
    "## Clear twins (from raw values)",
    "",
    "| Pair | Why |",
    "| --- | --- |",
    "| `composition` ↔ `composition` | 511 shared fibre tokens (cotton, polyester, wool…) |",
    "| `composition_detail` ↔ `composition` | 263 shared materials |",
    "| `colors` ↔ `color` | 167 shared color names (red, pink, navy…) — **best cross-name twin** |",
    "| `brand` ↔ `buyer` | zara, matalan, primark appear in both (ecom brand names live as PLM buyers) |",
    "| `section` ↔ `gender` | men / women / unisex overlap (Kids/Christmas only on ecom section) |",
    "| `subfamily` / `category` / `name` ↔ `garment_type` / `plm_sub_*` / AI tags | garment nouns: dress, jumper, socks… |",
    "",
    "## Weak / do not treat as same",
    "",
    "| Pair | Raw finding |",
    "| --- | --- |",
    "| `category` ↔ `category` | Almost **no** shared values (Clothes/Jackets vs sweater/cut sew…) |",
    "| stock counts ↔ knitting_time/weight | Fake overlap — both are numbers |",
    "| `construction` ↔ `composition` | Overlap is fibres leaking into ecom construction text, not true construction |",
    "",
    "JSON detail: `RAW_VALUE_OVERLAP_ECOM_PLMEX.json`",
]
out = Path(r"d:\Rasel-P\R&D-KG\data\generated\eval\RAW_VALUE_OVERLAP_ECOM_PLMEX.md")
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\nwrote", out)
