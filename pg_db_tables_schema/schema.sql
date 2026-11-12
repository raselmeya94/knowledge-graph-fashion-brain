-- Fashion Brain DB schema — CONFIRMED against post-enrich dumps
--   Zara    man-tshirts 20261029_091620 — 265/265 enriched (100%)
--   Primark sweaters    20261029_111301 — 244/266 enriched (91.7%)
-- See db/CROSS_CHECK.txt for fill-rate details.
--
-- Tables:
--   1) master_products   — shared cols both brands fill after enrich
--   2) zara_products     — full 65-col dump + zara ids
--   3) primark_products  — full dump + primark_id
--   3b) matalan_products — full dump + matalan_id
--   3c) c_and_a_products — full dump + c_and_a_id (brand slug "c-and-a")
--   3d) walmart_products — full dump + walmart_id
--   3e) calvin_klein_products — full dump + calvin_klein_id (brand slug "calvin-klein")
--   4) *_colorways       — colorway_tiles[] child rows

-- ===========================================================================
-- 1) MASTER (shared) — confirmed both-brand ≥80% after enrich
-- ===========================================================================
CREATE TABLE IF NOT EXISTS master_products (
    master_id                TEXT NOT NULL,
    brand                    TEXT NOT NULL,          -- 'zara' | 'primark'
    catalog_product_ids      JSONB NOT NULL DEFAULT '[]',
    seo_product_id           TEXT,
    product_ids              JSONB NOT NULL DEFAULT '[]',
    primary_product_id       TEXT,
    reference                TEXT,
    display_reference        TEXT,
    name                     TEXT,
    description              TEXT,
    composition              TEXT,
    composition_detail       JSONB NOT NULL DEFAULT '[]',
    construction             TEXT,
    -- nullable: Primark-heavy but useful on master when present
    care                     TEXT,
    fit                      TEXT,
    country_of_origin        TEXT,
    section                  TEXT,
    category                 TEXT,
    family                   TEXT,
    subfamily                TEXT,
    category_path            TEXT,                 -- Primark-heavy, null on Zara
    availability             TEXT,
    in_stock_count           INTEGER,              -- SKU/size units in stock
    out_stock_count          INTEGER,              -- SKU/size units out of stock
    enriched_from            TEXT,
    pdp_enriched             BOOLEAN NOT NULL DEFAULT FALSE,
    currency                 TEXT,
    price                    TEXT,
    original_price           TEXT,                 -- sparse both brands
    price_min                TEXT,
    price_max                TEXT,
    prices_seen              JSONB NOT NULL DEFAULT '[]',
    image                    TEXT,
    images                   JSONB NOT NULL DEFAULT '[]',
    image_count              INTEGER NOT NULL DEFAULT 0,
    colors                   JSONB NOT NULL DEFAULT '[]',
    color_count              INTEGER NOT NULL DEFAULT 0,
    colorway_tile_count      INTEGER NOT NULL DEFAULT 0,
    product_url              TEXT,
    product_url_with_variant TEXT,
    product_urls             JSONB NOT NULL DEFAULT '[]',
    source_url               TEXT,
    domain                   TEXT,
    catalog_id               TEXT,
    scraped_at               TIMESTAMPTZ,
    scraped_date             DATE,
    scraped_time             TEXT,
    scrape_job_id            TEXT,
    content_hash             TEXT,
    colorway_hash            TEXT,
    last_changed_at          TIMESTAMPTZ,
    created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (brand, master_id)
);

CREATE INDEX IF NOT EXISTS idx_master_brand ON master_products (brand);
CREATE INDEX IF NOT EXISTS idx_master_seo ON master_products (seo_product_id);
CREATE INDEX IF NOT EXISTS idx_master_scraped_at ON master_products (scraped_at);
CREATE INDEX IF NOT EXISTS idx_master_catalog ON master_products (catalog_id);


-- ===========================================================================
-- 2) ZARA dump (individual)
-- ===========================================================================
CREATE TABLE IF NOT EXISTS zara_products (
    master_id                TEXT PRIMARY KEY,
    catalog_product_ids      JSONB NOT NULL DEFAULT '[]',
    seo_product_id           TEXT,
    seo_keyword              TEXT,
    product_ids              JSONB NOT NULL DEFAULT '[]',
    primary_product_id       TEXT,
    zara_ids                 JSONB NOT NULL DEFAULT '[]',
    primary_zara_id          TEXT,
    reference                TEXT,
    display_reference        TEXT,
    discern_product_id       TEXT,
    name                     TEXT,
    description              TEXT,
    composition              TEXT,
    composition_detail       JSONB NOT NULL DEFAULT '[]',
    construction             TEXT,
    care                     JSONB,
    care_items               JSONB,
    fit                      TEXT,
    fit_details              TEXT,
    product_features         JSONB,
    sizes                    JSONB NOT NULL DEFAULT '[]',
    size_names               JSONB NOT NULL DEFAULT '[]',
    country_of_origin        TEXT,
    brand                    TEXT DEFAULT 'zara',
    section                  TEXT,
    category                 TEXT,
    family                   TEXT,
    subfamily                TEXT,
    category_path            TEXT,
    badges                   JSONB NOT NULL DEFAULT '[]',
    product_tags             JSONB NOT NULL DEFAULT '[]',
    availability             TEXT,
    in_stock_count           INTEGER,
    out_stock_count          INTEGER,
    kind                     TEXT,
    type                     TEXT,
    enriched_from            TEXT,
    pdp_enriched             BOOLEAN NOT NULL DEFAULT FALSE,
    currency                 TEXT,
    price                    TEXT,
    original_price           TEXT,
    price_min                TEXT,
    price_max                TEXT,
    prices_seen              JSONB NOT NULL DEFAULT '[]',
    image                    TEXT,
    images                   JSONB NOT NULL DEFAULT '[]',
    image_count              INTEGER NOT NULL DEFAULT 0,
    colors                   JSONB NOT NULL DEFAULT '[]',
    color_count              INTEGER NOT NULL DEFAULT 0,
    color_variants           JSONB NOT NULL DEFAULT '[]',
    colorway_tiles           JSONB NOT NULL DEFAULT '[]',
    colorway_tile_count      INTEGER NOT NULL DEFAULT 0,
    product_url              TEXT,
    product_url_with_variant TEXT,
    product_urls             JSONB NOT NULL DEFAULT '[]',
    source_url               TEXT,
    domain                   TEXT,
    catalog_id               TEXT,
    grid_positions           JSONB NOT NULL DEFAULT '[]',
    listing_page             INTEGER,
    listing_page_size        INTEGER,
    listing_is_last_page     BOOLEAN,
    scraped_at               TIMESTAMPTZ,
    scraped_date             DATE,
    scraped_time             TEXT,
    scrape_job_id            TEXT,
    created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at               TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_zara_primary_zara_id ON zara_products (primary_zara_id);
CREATE INDEX IF NOT EXISTS idx_zara_seo ON zara_products (seo_product_id);
CREATE INDEX IF NOT EXISTS idx_zara_catalog ON zara_products (catalog_id);
CREATE INDEX IF NOT EXISTS idx_zara_scraped_at ON zara_products (scraped_at);

CREATE TABLE IF NOT EXISTS zara_colorways (
    id                       BIGSERIAL PRIMARY KEY,
    master_id                TEXT NOT NULL REFERENCES zara_products (master_id) ON DELETE CASCADE,
    product_id               TEXT,
    color                    TEXT,
    color_id                 TEXT,
    price                    TEXT,
    original_price           TEXT,
    availability             TEXT,
    image                    TEXT,
    product_url              TEXT,
    grid_position            INTEGER
);

CREATE INDEX IF NOT EXISTS idx_zara_cw_master ON zara_colorways (master_id);


-- ===========================================================================
-- 3) PRIMARK dump (individual)
-- ===========================================================================
CREATE TABLE IF NOT EXISTS primark_products (
    master_id                TEXT PRIMARY KEY,
    catalog_product_ids      JSONB NOT NULL DEFAULT '[]',
    seo_product_id           TEXT,
    seo_keyword              TEXT,
    product_ids              JSONB NOT NULL DEFAULT '[]',
    primary_product_id       TEXT,
    primark_id               TEXT,                 -- Ref# (= seo_product_id)
    reference                TEXT,
    display_reference        TEXT,
    discern_product_id       TEXT,
    name                     TEXT,
    description              TEXT,
    composition              TEXT,
    composition_detail       JSONB NOT NULL DEFAULT '[]',
    construction             TEXT,
    care                     JSONB,                -- Primark-heavy
    care_items               JSONB,
    fit                      TEXT,                 -- Primark-heavy
    fit_details              TEXT,
    product_features         JSONB,
    sizes                    JSONB NOT NULL DEFAULT '[]',
    size_names               JSONB NOT NULL DEFAULT '[]',
    country_of_origin        TEXT,
    brand                    TEXT DEFAULT 'primark',
    section                  TEXT,
    category                 TEXT,
    family                   TEXT,
    subfamily                TEXT,
    category_path            TEXT,                 -- Primark-heavy
    badges                   JSONB NOT NULL DEFAULT '[]',
    product_tags             JSONB NOT NULL DEFAULT '[]',
    availability             TEXT,
    in_stock_count           INTEGER,
    out_stock_count          INTEGER,
    kind                     TEXT,
    type                     TEXT,
    enriched_from            TEXT,
    pdp_enriched             BOOLEAN NOT NULL DEFAULT FALSE,
    currency                 TEXT,
    price                    TEXT,
    original_price           TEXT,
    price_min                TEXT,
    price_max                TEXT,
    prices_seen              JSONB NOT NULL DEFAULT '[]',
    image                    TEXT,
    images                   JSONB NOT NULL DEFAULT '[]',
    image_count              INTEGER NOT NULL DEFAULT 0,
    colors                   JSONB NOT NULL DEFAULT '[]',
    color_count              INTEGER NOT NULL DEFAULT 0,
    color_variants           JSONB NOT NULL DEFAULT '[]',
    colorway_tiles           JSONB NOT NULL DEFAULT '[]',
    colorway_tile_count      INTEGER NOT NULL DEFAULT 0,
    product_url              TEXT,
    product_url_with_variant TEXT,
    product_urls             JSONB NOT NULL DEFAULT '[]',
    source_url               TEXT,
    domain                   TEXT,
    catalog_id               TEXT,
    grid_positions           JSONB NOT NULL DEFAULT '[]',
    listing_page             INTEGER,
    listing_page_size        INTEGER,
    listing_is_last_page     BOOLEAN,
    scraped_at               TIMESTAMPTZ,
    scraped_date             DATE,
    scraped_time             TEXT,
    scrape_job_id            TEXT,
    created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at               TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_primark_seo ON primark_products (seo_product_id);
CREATE INDEX IF NOT EXISTS idx_primark_id ON primark_products (primark_id);
CREATE INDEX IF NOT EXISTS idx_primark_catalog ON primark_products (catalog_id);
CREATE INDEX IF NOT EXISTS idx_primark_scraped_at ON primark_products (scraped_at);

CREATE TABLE IF NOT EXISTS primark_colorways (
    id                       BIGSERIAL PRIMARY KEY,
    master_id                TEXT NOT NULL REFERENCES primark_products (master_id) ON DELETE CASCADE,
    product_id               TEXT,
    color                    TEXT,
    color_id                 TEXT,
    price                    TEXT,
    original_price           TEXT,
    availability             TEXT,
    image                    TEXT,
    product_url              TEXT,
    grid_position            INTEGER
);

CREATE INDEX IF NOT EXISTS idx_primark_cw_master ON primark_colorways (master_id);


-- ===========================================================================
-- 3b) MATALAN dump (individual)
-- ===========================================================================
CREATE TABLE IF NOT EXISTS matalan_products (
    master_id                TEXT PRIMARY KEY,
    catalog_product_ids      JSONB NOT NULL DEFAULT '[]',
    seo_product_id           TEXT,
    seo_keyword              TEXT,
    product_ids              JSONB NOT NULL DEFAULT '[]',
    primary_product_id       TEXT,
    matalan_id               TEXT,
    reference                TEXT,
    display_reference        TEXT,
    discern_product_id       TEXT,
    name                     TEXT,
    description              TEXT,
    composition              TEXT,
    composition_detail       JSONB NOT NULL DEFAULT '[]',
    construction             TEXT,
    care                     JSONB,
    care_items               JSONB,
    fit                      TEXT,
    fit_details              TEXT,
    product_features         JSONB,
    sizes                    JSONB NOT NULL DEFAULT '[]',
    size_names               JSONB NOT NULL DEFAULT '[]',
    country_of_origin        TEXT,
    brand                    TEXT DEFAULT 'matalan',
    section                  TEXT,
    category                 TEXT,
    family                   TEXT,
    subfamily                TEXT,
    category_path            TEXT,
    badges                   JSONB NOT NULL DEFAULT '[]',
    product_tags             JSONB NOT NULL DEFAULT '[]',
    availability             TEXT,
    in_stock_count           INTEGER,
    out_stock_count          INTEGER,
    kind                     TEXT,
    type                     TEXT,
    enriched_from            TEXT,
    pdp_enriched             BOOLEAN NOT NULL DEFAULT FALSE,
    currency                 TEXT,
    price                    TEXT,
    original_price           TEXT,
    price_min                TEXT,
    price_max                TEXT,
    prices_seen              JSONB NOT NULL DEFAULT '[]',
    image                    TEXT,
    images                   JSONB NOT NULL DEFAULT '[]',
    image_count              INTEGER NOT NULL DEFAULT 0,
    colors                   JSONB NOT NULL DEFAULT '[]',
    color_count              INTEGER NOT NULL DEFAULT 0,
    color_variants           JSONB NOT NULL DEFAULT '[]',
    colorway_tiles           JSONB NOT NULL DEFAULT '[]',
    colorway_tile_count      INTEGER NOT NULL DEFAULT 0,
    product_url              TEXT,
    product_url_with_variant TEXT,
    product_urls             JSONB NOT NULL DEFAULT '[]',
    source_url               TEXT,
    domain                   TEXT,
    catalog_id               TEXT,
    grid_positions           JSONB NOT NULL DEFAULT '[]',
    listing_page             INTEGER,
    listing_page_size        INTEGER,
    listing_is_last_page     BOOLEAN,
    scraped_at               TIMESTAMPTZ,
    scraped_date             DATE,
    scraped_time             TEXT,
    scrape_job_id            TEXT,
    created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at               TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_matalan_seo ON matalan_products (seo_product_id);
CREATE INDEX IF NOT EXISTS idx_matalan_id ON matalan_products (matalan_id);
CREATE INDEX IF NOT EXISTS idx_matalan_catalog ON matalan_products (catalog_id);
CREATE INDEX IF NOT EXISTS idx_matalan_scraped_at ON matalan_products (scraped_at);

CREATE TABLE IF NOT EXISTS matalan_colorways (
    id                       BIGSERIAL PRIMARY KEY,
    master_id                TEXT NOT NULL REFERENCES matalan_products (master_id) ON DELETE CASCADE,
    product_id               TEXT,
    color                    TEXT,
    color_id                 TEXT,
    price                    TEXT,
    original_price           TEXT,
    availability             TEXT,
    image                    TEXT,
    product_url              TEXT,
    grid_position            INTEGER
);

CREATE INDEX IF NOT EXISTS idx_matalan_cw_master ON matalan_colorways (master_id);


-- ===========================================================================
-- 3c) C&A dump (individual) — brand slug "c-and-a", SQL prefix c_and_a
-- ===========================================================================
CREATE TABLE IF NOT EXISTS c_and_a_products (
    master_id                TEXT PRIMARY KEY,
    catalog_product_ids      JSONB NOT NULL DEFAULT '[]',
    seo_product_id           TEXT,
    seo_keyword              TEXT,
    product_ids              JSONB NOT NULL DEFAULT '[]',
    primary_product_id       TEXT,
    c_and_a_id               TEXT,
    reference                TEXT,
    display_reference        TEXT,
    discern_product_id       TEXT,
    name                     TEXT,
    description              TEXT,
    composition              TEXT,
    composition_detail       JSONB NOT NULL DEFAULT '[]',
    construction             TEXT,
    care                     JSONB,
    care_items               JSONB,
    fit                      TEXT,
    fit_details              TEXT,
    product_features         JSONB,
    sizes                    JSONB NOT NULL DEFAULT '[]',
    size_names               JSONB NOT NULL DEFAULT '[]',
    country_of_origin        TEXT,
    brand                    TEXT DEFAULT 'c-and-a',
    section                  TEXT,
    category                 TEXT,
    family                   TEXT,
    subfamily                TEXT,
    category_path            TEXT,
    badges                   JSONB NOT NULL DEFAULT '[]',
    product_tags             JSONB NOT NULL DEFAULT '[]',
    availability             TEXT,
    in_stock_count           INTEGER,
    out_stock_count          INTEGER,
    kind                     TEXT,
    type                     TEXT,
    enriched_from            TEXT,
    pdp_enriched             BOOLEAN NOT NULL DEFAULT FALSE,
    currency                 TEXT,
    price                    TEXT,
    original_price           TEXT,
    price_min                TEXT,
    price_max                TEXT,
    prices_seen              JSONB NOT NULL DEFAULT '[]',
    image                    TEXT,
    images                   JSONB NOT NULL DEFAULT '[]',
    image_count              INTEGER NOT NULL DEFAULT 0,
    colors                   JSONB NOT NULL DEFAULT '[]',
    color_count              INTEGER NOT NULL DEFAULT 0,
    color_variants           JSONB NOT NULL DEFAULT '[]',
    colorway_tiles           JSONB NOT NULL DEFAULT '[]',
    colorway_tile_count      INTEGER NOT NULL DEFAULT 0,
    product_url              TEXT,
    product_url_with_variant TEXT,
    product_urls             JSONB NOT NULL DEFAULT '[]',
    source_url               TEXT,
    domain                   TEXT,
    catalog_id               TEXT,
    grid_positions           JSONB NOT NULL DEFAULT '[]',
    listing_page             INTEGER,
    listing_page_size        INTEGER,
    listing_is_last_page     BOOLEAN,
    scraped_at               TIMESTAMPTZ,
    scraped_date             DATE,
    scraped_time             TEXT,
    scrape_job_id            TEXT,
    created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at               TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_c_and_a_seo ON c_and_a_products (seo_product_id);
CREATE INDEX IF NOT EXISTS idx_c_and_a_id ON c_and_a_products (c_and_a_id);
CREATE INDEX IF NOT EXISTS idx_c_and_a_catalog ON c_and_a_products (catalog_id);
CREATE INDEX IF NOT EXISTS idx_c_and_a_scraped_at ON c_and_a_products (scraped_at);

CREATE TABLE IF NOT EXISTS c_and_a_colorways (
    id                       BIGSERIAL PRIMARY KEY,
    master_id                TEXT NOT NULL REFERENCES c_and_a_products (master_id) ON DELETE CASCADE,
    product_id               TEXT,
    color                    TEXT,
    color_id                 TEXT,
    price                    TEXT,
    original_price           TEXT,
    availability             TEXT,
    image                    TEXT,
    product_url              TEXT,
    grid_position            INTEGER
);

CREATE INDEX IF NOT EXISTS idx_c_and_a_cw_master ON c_and_a_colorways (master_id);


-- ===========================================================================
-- 3d) WALMART dump (individual) - brand slug "walmart"
-- ===========================================================================
CREATE TABLE IF NOT EXISTS walmart_products (
    master_id                TEXT PRIMARY KEY,
    catalog_product_ids      JSONB NOT NULL DEFAULT '[]',
    seo_product_id           TEXT,
    seo_keyword              TEXT,
    product_ids              JSONB NOT NULL DEFAULT '[]',
    primary_product_id       TEXT,
    walmart_id               TEXT,
    reference                TEXT,
    display_reference        TEXT,
    discern_product_id       TEXT,
    name                     TEXT,
    description              TEXT,
    description_short        TEXT,
    description_long         TEXT,
    description_length       INTEGER,
    composition              TEXT,
    composition_detail       JSONB NOT NULL DEFAULT '[]',
    construction             TEXT,
    care                     JSONB,
    care_items               JSONB,
    fit                      TEXT,
    fit_details              TEXT,
    product_features         JSONB,
    specifications           JSONB,              -- full More details panel
    sizes                    JSONB NOT NULL DEFAULT '[]',
    size_names               JSONB NOT NULL DEFAULT '[]',
    country_of_origin        TEXT,
    brand                    TEXT DEFAULT 'walmart',
    product_brand            TEXT,               -- label brand from PDP (e.g. Gerber)
    gender                   TEXT,
    seller                   TEXT,
    seller_type              TEXT,
    upc                      TEXT,
    model                    TEXT,
    pdp_product_id           TEXT,               -- internal Walmart product.id
    sku                      TEXT,
    section                  TEXT,
    category                 TEXT,
    family                   TEXT,
    subfamily                TEXT,
    category_path            TEXT,
    walmart_category_path    TEXT,
    badges                   JSONB NOT NULL DEFAULT '[]',
    product_tags             JSONB NOT NULL DEFAULT '[]',
    availability             TEXT,
    in_stock_count           INTEGER,
    out_stock_count          INTEGER,
    kind                     TEXT,
    type                     TEXT,
    rating                   DOUBLE PRECISION,
    review_count             INTEGER,
    enriched_from            TEXT,
    pdp_enriched             BOOLEAN NOT NULL DEFAULT FALSE,
    sponsored                BOOLEAN,
    currency                 TEXT,
    price                    TEXT,
    original_price           TEXT,
    price_min                TEXT,
    price_max                TEXT,
    prices_seen              JSONB NOT NULL DEFAULT '[]',
    savings                  TEXT,
    original_price_max       TEXT,
    image                    TEXT,
    images                   JSONB NOT NULL DEFAULT '[]',
    image_count              INTEGER NOT NULL DEFAULT 0,
    colors                   JSONB NOT NULL DEFAULT '[]',
    color_count              INTEGER NOT NULL DEFAULT 0,
    color_variants           JSONB NOT NULL DEFAULT '[]',
    colorway_tiles           JSONB NOT NULL DEFAULT '[]',
    colorway_tile_count      INTEGER NOT NULL DEFAULT 0,
    selected_color           TEXT,
    selected_size            TEXT,
    variant_offers           JSONB,              -- full color x size matrix
    variant_count            INTEGER,
    product_url              TEXT,
    product_url_with_variant TEXT,
    product_urls             JSONB NOT NULL DEFAULT '[]',
    source_url               TEXT,
    domain                   TEXT,
    catalog_id               TEXT,
    grid_positions           JSONB NOT NULL DEFAULT '[]',
    listing_page             INTEGER,
    listing_page_size        INTEGER,
    listing_is_last_page     BOOLEAN,
    scraped_at               TIMESTAMPTZ,
    scraped_date             DATE,
    scraped_time             TEXT,
    scrape_job_id            TEXT,
    created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at               TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_walmart_seo ON walmart_products (seo_product_id);
CREATE INDEX IF NOT EXISTS idx_walmart_id ON walmart_products (walmart_id);
CREATE INDEX IF NOT EXISTS idx_walmart_catalog ON walmart_products (catalog_id);
CREATE INDEX IF NOT EXISTS idx_walmart_scraped_at ON walmart_products (scraped_at);

CREATE TABLE IF NOT EXISTS walmart_colorways (
    id                       BIGSERIAL PRIMARY KEY,
    master_id                TEXT NOT NULL REFERENCES walmart_products (master_id) ON DELETE CASCADE,
    product_id               TEXT,
    color                    TEXT,
    color_id                 TEXT,
    price                    TEXT,
    original_price           TEXT,
    availability             TEXT,
    image                    TEXT,
    product_url              TEXT,
    grid_position            INTEGER
);

CREATE INDEX IF NOT EXISTS idx_walmart_cw_master ON walmart_colorways (master_id);


-- ===========================================================================
-- 3e) CALVIN KLEIN dump (individual) - brand slug "calvin-klein"
-- ===========================================================================
CREATE TABLE IF NOT EXISTS calvin_klein_products (
    master_id                TEXT PRIMARY KEY,
    catalog_product_ids      JSONB NOT NULL DEFAULT '[]',
    seo_product_id           TEXT,
    seo_keyword              TEXT,
    product_ids              JSONB NOT NULL DEFAULT '[]',
    primary_product_id       TEXT,
    calvin_klein_id          TEXT,
    reference                TEXT,
    display_reference        TEXT,
    discern_product_id       TEXT,
    name                     TEXT,
    description              TEXT,
    composition              TEXT,
    composition_detail       JSONB NOT NULL DEFAULT '[]',
    construction             TEXT,
    care                     JSONB,
    care_items               JSONB,
    fit                      TEXT,
    fit_details              TEXT,
    product_features         JSONB,
    sizes                    JSONB NOT NULL DEFAULT '[]',
    size_names               JSONB NOT NULL DEFAULT '[]',
    country_of_origin        TEXT,
    brand                    TEXT DEFAULT 'calvin-klein',
    section                  TEXT,
    category                 TEXT,
    family                   TEXT,
    subfamily                TEXT,
    category_path            TEXT,
    badges                   JSONB NOT NULL DEFAULT '[]',
    product_tags             JSONB NOT NULL DEFAULT '[]',
    availability             TEXT,
    in_stock_count           INTEGER,
    out_stock_count          INTEGER,
    kind                     TEXT,
    type                     TEXT,
    enriched_from            TEXT,
    pdp_enriched             BOOLEAN NOT NULL DEFAULT FALSE,
    currency                 TEXT,
    price                    TEXT,
    original_price           TEXT,
    price_min                TEXT,
    price_max                TEXT,
    prices_seen              JSONB NOT NULL DEFAULT '[]',
    image                    TEXT,
    images                   JSONB NOT NULL DEFAULT '[]',
    image_count              INTEGER NOT NULL DEFAULT 0,
    colors                   JSONB NOT NULL DEFAULT '[]',
    color_count              INTEGER NOT NULL DEFAULT 0,
    color_variants           JSONB NOT NULL DEFAULT '[]',
    colorway_tiles           JSONB NOT NULL DEFAULT '[]',
    colorway_tile_count      INTEGER NOT NULL DEFAULT 0,
    product_url              TEXT,
    product_url_with_variant TEXT,
    product_urls             JSONB NOT NULL DEFAULT '[]',
    source_url               TEXT,
    domain                   TEXT,
    catalog_id               TEXT,
    grid_positions           JSONB NOT NULL DEFAULT '[]',
    listing_page             INTEGER,
    listing_page_size        INTEGER,
    listing_is_last_page     BOOLEAN,
    scraped_at               TIMESTAMPTZ,
    scraped_date             DATE,
    scraped_time             TEXT,
    scrape_job_id            TEXT,
    created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at               TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_calvin_klein_seo ON calvin_klein_products (seo_product_id);
CREATE INDEX IF NOT EXISTS idx_calvin_klein_id ON calvin_klein_products (calvin_klein_id);
CREATE INDEX IF NOT EXISTS idx_calvin_klein_catalog ON calvin_klein_products (catalog_id);
CREATE INDEX IF NOT EXISTS idx_calvin_klein_scraped_at ON calvin_klein_products (scraped_at);

CREATE TABLE IF NOT EXISTS calvin_klein_colorways (
    id                       BIGSERIAL PRIMARY KEY,
    master_id                TEXT NOT NULL REFERENCES calvin_klein_products (master_id) ON DELETE CASCADE,
    product_id               TEXT,
    color                    TEXT,
    color_id                 TEXT,
    price                    TEXT,
    original_price           TEXT,
    availability             TEXT,
    image                    TEXT,
    product_url              TEXT,
    grid_position            INTEGER
);

CREATE INDEX IF NOT EXISTS idx_calvin_klein_cw_master ON calvin_klein_colorways (master_id);


-- ===========================================================================
-- 4) CHANGE HISTORY (append-only, skip dump when content unchanged)
-- ===========================================================================
CREATE TABLE IF NOT EXISTS product_change_events (
    id                       BIGSERIAL PRIMARY KEY,
    brand                    TEXT NOT NULL,
    master_id                TEXT NOT NULL,
    product_url              TEXT,
    scrape_job_id            TEXT,
    catalog_id               TEXT,
    detected_at              TIMESTAMPTZ,
    scraped_at               TIMESTAMPTZ,
    change_type              TEXT,                 -- created | price | availability | content | mixed
    fields_changed           JSONB NOT NULL DEFAULT '[]',
    before_json              JSONB NOT NULL DEFAULT '{}',
    after_json               JSONB NOT NULL DEFAULT '{}',
    content_hash_before      TEXT,
    content_hash_after       TEXT
);

CREATE INDEX IF NOT EXISTS idx_change_events_master
    ON product_change_events (brand, master_id, detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_change_events_job
    ON product_change_events (scrape_job_id);


-- ===========================================================================
-- 5) SCRAPE JOBS (API source of truth after data/ file cleanup)
-- ===========================================================================
CREATE TABLE IF NOT EXISTS scrape_jobs (
    job_id                   TEXT PRIMARY KEY,
    job_type                 TEXT NOT NULL,
    status                   TEXT NOT NULL,
    url                      TEXT,
    catalog_id               TEXT,
    session_id               TEXT,
    stage                    TEXT,
    product_count            INTEGER,
    products_found           INTEGER,
    products_dump_total      INTEGER,
    products_dumped          INTEGER,
    products_created         INTEGER,
    products_updated         INTEGER,
    products_unchanged       INTEGER,
    products_dump_failed     INTEGER,
    pause_requested          BOOLEAN NOT NULL DEFAULT FALSE,
    cancel_requested         BOOLEAN NOT NULL DEFAULT FALSE,
    error                    TEXT,
    payload_json             JSONB NOT NULL DEFAULT '{}',
    result_json              JSONB,
    changes_summary_json     JSONB,
    created_at               TIMESTAMPTZ,
    updated_at               TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_scrape_jobs_status ON scrape_jobs (status);
CREATE INDEX IF NOT EXISTS idx_scrape_jobs_updated ON scrape_jobs (updated_at DESC);
