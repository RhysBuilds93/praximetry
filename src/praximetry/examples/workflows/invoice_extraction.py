"""Workflow 2: structured extraction (invoice text -> validated JSON).

A classic "expensive habit": frontier model + verbose schema prompt for what is
mostly deterministic extraction. Golden examples use the json_match scorer with
stage-level expected outputs, so the optimizer can prove a small model extracts
the same fields.

Try:
    python -m praximetry.examples.workflows.invoice_extraction
    praximetry eval --stage extract_invoice -m praximetry.examples.workflows.invoice_extraction --fail-under 0.9
    praximetry optimize --stage extract_invoice -m praximetry.examples.workflows.invoice_extraction
    praximetry apply --stage extract_invoice
"""

import json

import praximetry as px

from ._real import premium_model, real_chat

px.init(project="invoice-extraction")

SCHEMA = {
    "type": "object",
    "properties": {
        "vendor": {"type": "string", "description": "The company issuing the invoice"},
        "invoice_number": {"type": "string", "description": "Unique invoice identifier"},
        "total": {"type": "number", "description": "Grand total in the invoice currency"},
        "currency": {"type": "string", "description": "ISO 4217 code, e.g. USD"},
        "due_date": {"type": "string", "description": "ISO date the payment is due"},
    },
    "required": ["vendor", "invoice_number", "total", "currency"],
}


@px.stage("extract_invoice")
def extract_invoice(invoice_text: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a precise data-extraction engine. Return ONLY valid JSON "
            "matching the schema. Do not add commentary.",
        },
        {
            "role": "user",
            "content": f"JSON schema:\n{json.dumps(SCHEMA, indent=2)}\n\nInvoice text:\n{invoice_text}",
        },
    ]
    return real_chat(premium_model(), messages)


INVOICES = [
    "From: Globex Corp\nInvoice #: GX-1041\nTotal: $2,450.00\nDue: 2026-08-01",
    "Initech invoice number INT-77 amount due USD 149.99 due date: 2026-07-30",
    "From: Umbrella GmbH\nInvoice No. UM-2210\nTotal: €880.50\nDue: 2026-09-15",
]

if __name__ == "__main__":
    for inv in INVOICES:
        print(f"  {extract_invoice(inv)}")
    print("\nTraffic recorded. Try: praximetry-cloud detect")
