"""Product-spec response system must be buyer intelligence, not a mode dump."""

from fashion_brain.agents.product_spec import (
    SPECIFICATION_RESPONSE_SYSTEM,
    findings_are_product_specification,
)
from fashion_brain.agents._impl import response_messages
from fashion_brain.agents.contracts import Retrieved


def test_specification_response_system_is_intelligence_brief():
    text = SPECIFICATION_RESPONSE_SYSTEM.casefold()
    assert "product development" in text or "intelligence analyst" in text
    assert "pending plmex" in text
    assert "historically observed" in text or "historical" in text
    assert "not invent" in text or "never invent" in text
    assert "final product specification" in text or "json" in text
    # Must not treat mode pick as existing SKU.
    assert "existing" in text and ("plmex" in text or "catalog" in text)


def test_findings_detector_accepts_proposed_sheet_markers():
    blob = (
        "# Winter Product Specification — Primark\n\n"
        "## 1. User Requirements\n"
        "- **Brand:** Primark\n\n"
        "## 2. Historical Data Analysis\n"
        "### Attribute Analysis\n"
        "| Attribute | Leading Value | Count | Alternative Values |\n"
        "| Category | Sweater | 10 | Cardigan (4) |\n\n"
        "## 4. Final Product Specification\n"
        "Product specification:\n"
        "Proposed new product specification (NOT an existing PLM/catalog record):\n"
        "Product identity: Pending PLMEX creation/assignment\n"
        "```json\n"
        '{"material": "Wool", "approval_status": "Draft - Pending Buyer Review"}\n'
        "```\n"
    )
    assert findings_are_product_specification(blob)


def test_response_messages_use_intelligence_task_for_spec_findings():
    retrieved = Retrieved(
        text=(
            "# Winter Product Specification — Primark\n\n"
            "## 1. User Requirements\n- **Brand:** Primark\n\n"
            "## 2. Historical Data Analysis\n"
            "### Attribute Analysis\n"
            "| Attribute | Leading Value | Count | Alternative Values |\n"
            "| Material | Wool | 7 | Cotton (3) |\n\n"
            "## 4. Final Product Specification\n"
            "Product specification:\n"
            "Proposed new product specification (NOT an existing PLM/catalog record):\n"
            "Product identity: Pending PLMEX creation/assignment\n"
            "```json\n"
            '{"material": "Wool", "approval_status": "Draft - Pending Buyer Review"}\n'
            "```\n"
        ),
        evidence=[],
        caveats=[],
        unavailable=[],
    )
    messages = response_messages(
        "Based on typical Primark sweaters, sketch a product specification outline.",
        retrieved,
    )
    assert messages[0]["role"] == "system"
    assert "Product Development Intelligence" in messages[0]["content"]
    user = messages[1]["content"]
    assert "PROPOSED new product specification" in user or "Final Product Specification" in user
    assert "Do not invent IDs" in user or "Do not invent" in user
