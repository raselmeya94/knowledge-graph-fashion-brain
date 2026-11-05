# Data root

All KG **data** lives here — separate from Python under `knowledge_graph/`.

```text
data/
  schemas/                 Ontologies + ecom mapping (committed seeds)
  raw/                     PLM files + source_registry.json (optional)
  generated/               Runtime only (gitignored except .gitkeep)
  config/                  Optional entity review decisions
```

See `knowledge_graph/FRESH_DEVICE.md` for clone → run on a new machine.
