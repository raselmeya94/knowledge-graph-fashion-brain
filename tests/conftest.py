from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pytest

from knowledge_graph.adapters.products import RAW_COLUMNS
from knowledge_graph.config import DATA_ROOT, KG_ROOT, Settings
from knowledge_graph.hashing import sha256_bytes
from knowledge_graph.pipeline.service import Platform

ROOT = KG_ROOT
RAW = DATA_ROOT / "raw"

SAMPLE_ROWS = [
    {
        "Code": "1000000001",
        "Composition": "100%Cotton",
        "Category": "Tee",
        "Garment Type": "Tee",
        "color": "Red",
        "Season Year": "2025",
        "Style Reference": "TEE-001",
        "Buyer": "BESTSELLER",
        "Brand": "Northwind Mills",
    },
    {
        "Code": "1000000002",
        "Composition": "100%Cotton",
        "Category": "Shirt",
        "Season Year": "2024",
        "Style Reference": "SHIRT-002",
        "Buyer": "OTHER",
    },
    {
        "Code": "1000000003",
        "Composition": "Cotton Blend",
        "Category": "Sweater",
        "Garment Type": "Sweater",
        "color": "Red",
        "Gender": "Women",
        "Season Year": "2025",
        "Style Reference": "BLEND-003",
        "Buyer": "BESTSELLER",
    },
    {
        "Code": "1000000004",
        "Composition": "100%Elastane",
        "Category": "Swim",
        "Season Year": "2025",
        "Style Reference": "SWIM-004",
        "Buyer": "BESTSELLER",
    },
    {
        "Code": "1000000005",
        "Composition": "100%Cotton",
        "Category": "Sweater",
        "Garment Type": "Dress",
        "color": "蓝色",
        "Season Year": "2025",
        "Style Reference": "OG RED KNITTED DRESS",
        "Buyer": "BESTSELLER",
    },
    {
        "Code": "1000000006",
        "Composition": "100%Cotton",
        "Category": "Sweater",
        "Garment Type": "Jumper",
        "color": "红色",
        "Gender": "Men",
        "Season Year": "2025",
        "Style Reference": "KNIT-006",
        "Buyer": "BESTSELLER",
    },
]


def write_sample_catalog(root: Path) -> Path:
    """Small catalog that uses the real CSV headers. Tests do not load the full export."""

    source = root / "own_products"
    source.mkdir(parents=True)
    shutil.copy(RAW / "source_registry.json", root / "source_registry.json")
    csv_path = source / "products_sample.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in SAMPLE_ROWS:
            writer.writerow({column: row.get(column, "") for column in RAW_COLUMNS})
    manifest = {
        "source_id": "own_products",
        "records": [
            {
                "record_id": "*",
                "path": "products_sample.csv",
                "format": "csv",
                "hash_method": "file",
                "content_hash": sha256_bytes(csv_path.read_bytes()),
                "source_timestamp": None,
                "ingestion_status": "registered",
                "synthetic": False,
            }
        ],
    }
    (source / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return root


@pytest.fixture(scope="session")
def demo(tmp_path_factory) -> Platform:
    raw_dir = write_sample_catalog(tmp_path_factory.mktemp("sample-raw"))
    settings = Settings(
        env="test",
        raw_data_dir=raw_dir,
        generated_dir=tmp_path_factory.mktemp("generated"),
        graph_backend="memory",
        api_token="test-token",
        extraction_mode="deterministic",
        answer_mode="template",
        model_provider="none",
        sql_enabled=False,
        sql_dsn="",
        log_json=False,
        kg_url="",
    )
    platform = Platform(settings)
    platform.rebuild()
    return platform


@pytest.fixture()
def copied_raw(tmp_path: Path) -> Path:
    return write_sample_catalog(tmp_path / "raw_data")


def make_platform(raw_dir: Path, generated: Path, **overrides) -> Platform:
    settings = Settings(
        env=overrides.pop("env", "test"),
        raw_data_dir=raw_dir,
        generated_dir=generated,
        graph_backend=overrides.pop("graph_backend", "memory"),
        api_token=overrides.pop("api_token", "test-token"),
        extraction_mode="deterministic",
        answer_mode="template",
        model_provider="none",
        sql_enabled=False,
        sql_dsn="",
        log_json=False,
        kg_url=overrides.pop("kg_url", ""),
        **overrides,
    )
    return Platform(settings)
