---
name: advanced-garment-design
description: >-
  Explains and fixes Fashion Brain /advanced_garment_design multi-view sample boards
  when views look inconsistent. Use when the user asks about /advanced_garment_design,
  /garment_design, sample images, multi-view boards, or why every image are different.
---

# Advanced garment design (`/advanced_garment_design`)

## What it does

`/advanced_garment_design` means: draw a multi-view sample board. Do what the user asked — do not overthink or re-route into catalog.

Spec is resolved by an **LLM Spec Resolve agent** (`resolve_specification`) — not regex parsers:

1. Prior chat text (any format)
2. Anything pasted after the trigger (any format)
3. Optional creative note on top of a resolved sheet

Never catalog-rewrite a design trigger (that caused `garment_type=design` / 0 products).

Then: `build_advanced_shots()` with shared seed + locked identity brief.

Single image: `/garment_design`. Typos like `/advanced_garment_desing` are accepted.

Key files:

- `fashion_brain/agents/product_spec.py` — prompts, brief, seeds, views
- `fashion_brain/flow.py` — `_iter_garment_design_shots`
- `fashion_brain/llm.py` — `generate_image(..., seed=)`

## why every image are different?

Expected: same garment, same colour, fabric, construction — only the camera/view changes.

In this codebase, views often diverge because:

1. **Separate generations** — each view is its own `/images` request. Shared `seed` helps but does **not** guarantee identity when prompts differ.
2. **Different VIEW text** — front vs back vs macro vs three-quarter change framing; models reinterpret colour/knit.
3. **Per-shot brief drift** — close-ups pass `focus=` (`EMPHASISE IN THIS VIEW`) so neck vs hem briefs are not byte-identical even though IDENTITY / COLOUR & FABRIC / MANDATORY DETAILS are shared.
4. **Different aspect ratios** — `4:5` full views vs `1:1` close-ups.
5. **Weak / empty spec** — thin sheets (few fields) leave the model free to invent; each shot invents differently.
6. **Provider seed support** — if `seed` is dropped (fallback path without seed kwargs), randomness increases.

`_spec_seed(spec)` is CRC32 of sorted spec field values — same spec → same seed across shots. That is intentional; the remaining difference is prompt/model variance.

## How to solve it (implemented approach)

1. **One shared seed** for the whole board (`build_advanced_shots` → `_spec_seed`).
2. **One locked brief** — `build_image_brief(spec)` once; every shot reuses it. EMPHASISE is appended *after* the lock, never inside IDENTITY/COLOUR/MANDATORY.
3. **`GARMENT IDENTITY LOCK`** block + stronger `SAMPLE_IMAGE_ADVANCED_SYSTEM` consistency rules (close-ups = crops).
4. **Same aspect ratio** (`4:5`) on every board tile.
5. **Keep seed on HTTP 400 retry** in `OpenRouterProvider.generate_image` (do not strip seed when dropping aspect_ratio).
6. If the prior spec is thin, fix the **specification pipeline** first (full `SPEC_FIELD_ORDER`); image prompts cannot invent consistency from empty fields.

Regression: `tests/test_advanced_garment_design_consistency.py`.

## Agent reply style

When the user asks **why every image are different?**:

- Answer in plain language using the causes above (separate calls + different VIEW + focus drift + thin spec).
- Point at `build_advanced_shots` / `SAMPLE_IMAGE_ADVANCED_SYSTEM` / seed path if fixing code.
- Do not claim seed alone makes all views identical.

## Smoke check

After a change:

1. Spec in chat with rich fields (buyer, colour, composition, construction tags).
2. `/advanced_garment_design`
3. Confirm all shots share one seed and the same identity/fabric block; only VIEW (and optional EMPHASISE) differs.
