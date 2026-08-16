"""
parse_fhir_bundle.py

Reads a real FHIR Bundle back into our internal shapes (vitals rows, lab
rows, note rows, medication mentions), the reverse direction of
generate_fhir_export.py. Genuine parsing: walks the actual resource array,
matches on resourceType and LOINC codes, decodes base64 note content.
"""

import json
import base64


def parse_fhir_bundle(path):
    with open(path) as f:
        bundle = json.load(f)
    resources = [e["resource"] for e in bundle["entry"]]

    patient = next(r for r in resources if r["resourceType"] == "Patient")
    encounter = next(r for r in resources if r["resourceType"] == "Encounter")

    vitals, labs, notes, meds = [], [], [], []
    for r in resources:
        rtype = r["resourceType"]
        if rtype == "Observation":
            category = r["category"][0]["coding"][0]["code"]
            code = r["code"]["coding"][0]
            row = dict(
                subject_id=int(patient["id"]),
                loinc_code=code["code"], label=code["display"],
                charttime=r.get("effectiveDateTime"),
                value=(r.get("valueQuantity") or {}).get("value"),
                unit=(r.get("valueQuantity") or {}).get("unit"),
            )
            if category == "vital-signs":
                vitals.append(row)
            elif category == "laboratory":
                row["storetime"] = r.get("issued")
                row["flag"] = (r.get("interpretation") or [{}])[0].get("text")
                labs.append(row)
        elif rtype == "DocumentReference":
            data_b64 = r["content"][0]["attachment"]["data"]
            text = base64.b64decode(data_b64).decode("utf-8")
            notes.append(dict(
                subject_id=int(patient["id"]), note_id=r["id"],
                note_type=r["type"]["text"], author_type=r["author"][0]["type"],
                event_charttime=r["context"]["period"]["start"], storetime=r["date"],
                text=text,
            ))
        elif rtype == "MedicationStatement":
            meds.append(dict(
                subject_id=int(patient["id"]),
                medication=r["medicationCodeableConcept"]["text"],
                dosage=r.get("dosage", [{}])[0].get("text"),
                reason=(r.get("reasonCode") or [{}])[0].get("text"),
                effectiveDateTime=r.get("effectiveDateTime"),
            ))

    provenance = dict(
        format="HL7 FHIR R4 Bundle",
        source_system="EHR export",
        raw_source_file=str(path),
    )
    return dict(
        patient=dict(subject_id=int(patient["id"]), gender=patient["gender"], birthDate=patient["birthDate"]),
        encounter=dict(hadm_id=int(encounter["id"]), period=encounter["period"], careunit=encounter["serviceProvider"]["display"]),
        vitals=vitals, labs=labs, notes=notes, medications=meds,
        provenance=provenance,
    )


if __name__ == "__main__":
    import argparse
    import glob

    parser = argparse.ArgumentParser(description="Quick sanity check: parse every real FHIR bundle found.")
    parser.add_argument("--data-dir", default=".",
                         help="Folder containing XML_JSON/fhir_export/. Run with --data-dir ./data "
                              "if you're running this from the Tutorial/ root.")
    args = parser.parse_args()

    for f in sorted(glob.glob(f"{args.data_dir}/XML_JSON/fhir_export/*.json")):
        parsed = parse_fhir_bundle(f)
        print(f"{f.split('/')[-1]:35s} vitals={len(parsed['vitals']):>4} labs={len(parsed['labs']):>3} "
              f"notes={len(parsed['notes']):>3} meds={len(parsed['medications'])}")
