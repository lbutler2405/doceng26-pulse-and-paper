"""
amendment_schema_demo.py

The "break it on purpose" moment for Block 1: build one real patient's
document (the same build_documents() used everywhere else), take a genuine
amendment out of it, and show document_schema.json accepting the well-formed
version and rejecting a deliberately broken one, an amendment missing
amends_note_id, an orphaned correction that doesn't say what it corrects.

This runs against the REAL schema and a REAL assembled document, not a
standalone fragment, now that document_schema.json and
document_representation.py both carry the amendment fields.

Run directly for the demo:

    python amendment_schema_demo.py --data-dir .
"""

import argparse
import copy
import json

from jsonschema import validate, ValidationError

from document_representation import build_documents


def find_amendment_note(docs):
    """First (document, note) pair anywhere in the corpus where the note is
    a genuine amendment."""
    for doc in docs.values():
        for note in doc["evidence_streams"]["delayed_narrative"]["notes"]:
            if note.get("is_amendment"):
                return doc, note
    return None, None


def run_demo(data_dir="."):
    with open(f"{data_dir}/document_schema.json") as f:
        schema = json.load(f)

    docs = build_documents(data_dir)
    doc, amendment_note = find_amendment_note(docs)
    if doc is None:
        print("No amendment notes found, run generate_amendments.py first.")
        return

    print(f"Using patient {doc['subject_id']}'s real amendment note {amendment_note['note_id']}, "
          f"which corrects {amendment_note['amends_note_id']}.\n")

    print("Validating the whole document as build_documents() actually produced it...")
    try:
        validate(instance=doc, schema=schema)
        print("  PASSED: document validates.\n")
    except ValidationError as e:
        print(f"  FAILED (unexpected): {e.message}\n")

    print("Now deliberately orphaning that amendment, removing amends_note_id, "
          "the one thing that says what it's correcting...")
    broken_doc = copy.deepcopy(doc)
    for note in broken_doc["evidence_streams"]["delayed_narrative"]["notes"]:
        if note.get("is_amendment") and note["note_id"] == amendment_note["note_id"]:
            del note["amends_note_id"]
    try:
        validate(instance=broken_doc, schema=schema)
        print("  PASSED (this should not happen)\n")
    except ValidationError as e:
        print(f"  REJECTED, as it should be: {e.message}")
        print(f"  at: evidence_streams -> delayed_narrative -> notes -> "
              f"{'.'.join(str(p) for p in e.path)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=".")
    args = parser.parse_args()
    run_demo(args.data_dir)
