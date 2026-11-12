-- PLMEX products — single table from plmex_data/products_data.csv (43 cols)
-- Target DB example: CREATE DATABASE plmex_live;
-- Load: see COPY at bottom (or use psql \copy)

CREATE TABLE IF NOT EXISTS plmex_products (
    created_by                 TEXT,
    code                       TEXT PRIMARY KEY,
    composition                TEXT,
    category                   TEXT,
    plm_main_category          TEXT,
    plm_sub_category           TEXT,
    plm_sub_sub_category       TEXT,
    brand                      TEXT,
    unit                       TEXT,
    gauge                      TEXT,
    gender                     TEXT,
    sample_collection_type     TEXT,
    shipment_quantity          TEXT,
    shipment_date              TEXT,
    garment_type               TEXT,
    design_for                 TEXT,
    knit_machine_type          TEXT,
    knitting_time              TEXT,
    design_feature             TEXT,
    merchandiser               TEXT,
    department                 TEXT,
    shopping_department        TEXT,
    season_year                TEXT,
    designer                   TEXT,
    style_reference            TEXT,
    customer_style_reference   TEXT,
    style_feature              TEXT,
    color                      TEXT,
    weight                     TEXT,
    gsm                        TEXT,
    cuttable_width             TEXT,
    sample_making_date         TEXT,
    current_stock              INTEGER,
    current_location           TEXT,
    purchased_by               TEXT,
    created_at                 TEXT,
    updated_at                 TEXT,
    buyer                      TEXT,
    ai_event_tags              TEXT,
    ai_construction_tags       TEXT,
    ai_material_tags           TEXT,
    ai_finish_tags             TEXT,
    ai_special_tags            TEXT
);

CREATE INDEX IF NOT EXISTS idx_plmex_brand ON plmex_products (brand);
CREATE INDEX IF NOT EXISTS idx_plmex_buyer ON plmex_products (buyer);
CREATE INDEX IF NOT EXISTS idx_plmex_category ON plmex_products (category);
CREATE INDEX IF NOT EXISTS idx_plmex_garment_type ON plmex_products (garment_type);
CREATE INDEX IF NOT EXISTS idx_plmex_gender ON plmex_products (gender);
CREATE INDEX IF NOT EXISTS idx_plmex_season_year ON plmex_products (season_year);

-- ---------------------------------------------------------------------------
-- CSV → table (run after CREATE DATABASE plmex_live; \c plmex_live)
-- Header row must match column order below. Dates stay TEXT as in the CSV.
-- ---------------------------------------------------------------------------
-- \copy plmex_products (
--     created_by, code, composition, category,
--     plm_main_category, plm_sub_category, plm_sub_sub_category,
--     brand, unit, gauge, gender, sample_collection_type,
--     shipment_quantity, shipment_date, garment_type, design_for,
--     knit_machine_type, knitting_time, design_feature, merchandiser,
--     department, shopping_department, season_year, designer,
--     style_reference, customer_style_reference, style_feature, color,
--     weight, gsm, cuttable_width, sample_making_date,
--     current_stock, current_location, purchased_by, created_at, updated_at,
--     buyer, ai_event_tags, ai_construction_tags, ai_material_tags,
--     ai_finish_tags, ai_special_tags
-- ) FROM 'plmex_data/products_data.csv' WITH (FORMAT csv, HEADER true, ENCODING 'UTF8');
