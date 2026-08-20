"""
document_representation.py

Assembles the flat, relational synthetic_*.csv files into one self-contained,
schema-defined document object per admission, and demonstrates rendering a
human-readable chart summary back out of that structured document.

This is the piece that makes the tutorial actually about document
engineering rather than just tabular data science: CSVs are a relational
representation; what follows is a document representation, with embedded
provenance, validated against document_schema.json.

Two outputs:
  synthetic_documents.json   - one JSON object per hadm_id, all 75 bundled
  example_chart_<id>.md      - one admission rendered as a human-readable
                                document, generated FROM the structured form,
                                not from anything hand-written
"""

import json
import glob
import pandas as pd
from datetime import datetime

from parse_muse_xml import parse_muse_xml
from parse_fhir_bundle import parse_fhir_bundle

SCHEMA_VERSION = "1.0"
GENERATED_BY = "synthetic_generator.py"


def _iso(x):
    if pd.isna(x):
        return None
    if isinstance(x, str):
        return x
    return pd.Timestamp(x).strftime("%Y-%m-%dT%H:%M:%S")


def _discover_raw_sources(data_dir):
    """Finds whatever real MUSE XML / FHIR bundle files actually exist
    alongside the CSVs, under data_dir/XML_JSON/. Returns (fhir_by_subject,
    muse_by_ecg_id). Both are empty dicts if the corresponding folder isn't
    present, everything falls back to CSV-sourced construction cleanly
    either way."""
    fhir_by_subject = {}
    for path in glob.glob(f"{data_dir}/XML_JSON/fhir_export/*.json"):
        sid = int(path.split("_")[-1].replace(".json", ""))
        fhir_by_subject[sid] = path

    muse_by_ecg_id = {}
    for path in glob.glob(f"{data_dir}/XML_JSON/muse_xml/*.xml"):
        try:
            parsed = parse_muse_xml(path)
        except Exception:
            continue
        # match back to the CSV row this file was generated from, by subject
        # id + finding text (the file itself doesn't store our internal ecg_id)
        muse_by_ecg_id[path] = parsed
    return fhir_by_subject, muse_by_ecg_id


def build_documents(data_dir="."):
    patients = pd.read_csv(f"{data_dir}/synthetic_patients.csv", parse_dates=["admittime", "dischtime"])
    vitals = pd.read_csv(f"{data_dir}/synthetic_vitals_continuous.csv", parse_dates=["charttime"])
    labs = pd.read_csv(f"{data_dir}/synthetic_labs_intermittent.csv", parse_dates=["charttime", "storetime"])
    notes = pd.read_csv(f"{data_dir}/synthetic_notes.csv", parse_dates=["event_charttime", "storetime"])
    med_labels = pd.read_csv(f"{data_dir}/synthetic_medication_labels.csv")
    ecg = pd.read_csv(f"{data_dir}/synthetic_ecg_events.csv", parse_dates=["ecg_time", "report_time"])
    echo = pd.read_csv(f"{data_dir}/synthetic_echo_events.csv", parse_dates=["echo_time", "report_time"])

    fhir_by_subject, muse_parsed_by_path = _discover_raw_sources(data_dir)
    # ecg_id -> parsed MUSE dict, matched by subject_id + finding text (the
    # only shared keys between our CSV and a real file that doesn't know our
    # internal ids)
    muse_by_ecg_row = {}
    for path, parsed in muse_parsed_by_path.items():
        candidates = ecg[(ecg.subject_id == parsed["subject_id"]) & (ecg.finding == parsed["finding"])]
        for eid in candidates.ecg_id:
            muse_by_ecg_row[eid] = parsed

    CSV_PROVENANCE = dict(format="Tutorial CSV export", source_system="synthetic_generator.py", raw_source_file=None)

    documents = {}

    for prow in patients.itertuples():
        hadm_id = int(prow.hadm_id)
        subject_id = int(prow.subject_id)

        fhir_path = fhir_by_subject.get(subject_id)
        fhir_data = parse_fhir_bundle(fhir_path) if fhir_path else None

        pv = vitals[vitals.subject_id == subject_id]
        pl = labs[labs.subject_id == subject_id]
        pn = notes[notes.subject_id == subject_id].copy()
        pe = ecg[ecg.subject_id == subject_id]
        pec = echo[echo.subject_id == subject_id]

        note_objs = []
        if fhir_data:
            
            for n in fhir_data["notes"]:
                note_objs.append(dict(
                    note_id=n["note_id"], note_type=n["note_type"], author_type=n["author_type"],
                    event_charttime=n["event_charttime"], storetime=n["storetime"],
                    text=n["text"], history_reliability=None, related_event_id=None,
                    is_amendment=False, amends_note_id=None, amendment_reason=None,
                    medication_mentions=[],
                ))


            amendment_rows = pn[pn.get("is_amendment", 0) == 1] if "is_amendment" in pn.columns else pn.iloc[0:0]
            for n in amendment_rows.itertuples():
                med_spans = med_labels[med_labels.note_id == n.note_id]
                mentions = [
                    dict(
                        start_position=int(row["Start Position"]), end_position=int(row["End Position"]),
                        annotation=row["Annotation"], group=row["Group"], text=row["Text"],
                    )
                    for _, row in med_spans.iterrows()
                ] if len(med_spans) else []
                note_objs.append(dict(
                    note_id=n.note_id, note_type=n.note_type, author_type=n.author_type,
                    event_charttime=_iso(n.event_charttime), storetime=_iso(n.storetime),
                    text=n.text, history_reliability=None, related_event_id=None,
                    is_amendment=True, amends_note_id=n.amends_note_id, amendment_reason=n.amendment_reason,
                    medication_mentions=mentions,
                ))
        else:
            for n in pn.itertuples():
                med_spans = med_labels[med_labels.note_id == n.note_id]
                mentions = [
                    dict(
                        start_position=int(row["Start Position"]), end_position=int(row["End Position"]),
                        annotation=row["Annotation"], group=row["Group"],
                        text=row["Text"],
                    )
                    for _, row in med_spans.iterrows()
                ] if len(med_spans) else []

                is_amendment = bool(getattr(n, "is_amendment", 0)) if "is_amendment" in pn.columns else False
                amends_note_id = (n.amends_note_id if "amends_note_id" in pn.columns and pd.notna(n.amends_note_id)
                                   else None)
                amendment_reason = (n.amendment_reason if "amendment_reason" in pn.columns and pd.notna(n.amendment_reason)
                                     else None)

                note_objs.append(dict(
                    note_id=n.note_id, note_type=n.note_type, author_type=n.author_type,
                    event_charttime=_iso(n.event_charttime), storetime=_iso(n.storetime),
                    text=n.text,
                    history_reliability=n.history_reliability if pd.notna(n.history_reliability) else None,
                    related_event_id=n.related_event_id if ("related_event_id" in pn.columns and pd.notna(n.related_event_id)) else None,
                    is_amendment=is_amendment, amends_note_id=amends_note_id, amendment_reason=amendment_reason,
                    medication_mentions=mentions,
                ))

        if fhir_data:
            vitals_provenance = dict(format="HL7 FHIR R4 Bundle", source_system="EHR export", raw_source_file=fhir_path)
            labs_provenance = dict(vitals_provenance)
            notes_provenance = dict(vitals_provenance)
            vitals_readings = [
                dict(charttime=v["charttime"], heartrate=None, resprate=None, o2sat=None, sbp=None, dbp=None,
                     signal_quality="normal", loinc_code=v["loinc_code"], label=v["label"], value=v["value"])
                for v in fhir_data["vitals"]
            ]
            n_readings = len(vitals_readings)
            labs_draws = [
                dict(specimen_id=None, charttime=l["charttime"], storetime=l["storetime"],
                     label=l["label"], value=l["value"], valueuom=l["unit"], flag=l["flag"])
                for l in fhir_data["labs"]
            ]
            n_draws = len(labs_draws)
        else:
            vitals_provenance = CSV_PROVENANCE
            labs_provenance = CSV_PROVENANCE
            notes_provenance = CSV_PROVENANCE
            vitals_readings = [
                dict(charttime=_iso(r.charttime), heartrate=r.heartrate, resprate=r.resprate,
                     o2sat=r.o2sat, sbp=r.sbp, dbp=r.dbp, signal_quality=r.signal_quality)
                for r in pv.itertuples()
            ]
            n_readings = len(pv)
            labs_draws = [
                dict(specimen_id=int(r.specimen_id), charttime=_iso(r.charttime), storetime=_iso(r.storetime),
                     label=r.label, value=r.value, valueuom=r.valueuom,
                     flag=r.flag if pd.notna(r.flag) else None)
                for r in pl.itertuples()
            ]
            n_draws = pl.specimen_id.nunique()

        comorbidities = [] if prow.comorbidities == "none" else [c.strip() for c in prow.comorbidities.split(",")]

        doc = {
            "schema_version": SCHEMA_VERSION,
            "generated_by": GENERATED_BY,
            "hadm_id": hadm_id,
            "subject_id": subject_id,
            "admission": {
                "admittime": _iso(prow.admittime), "dischtime": _iso(prow.dischtime),
                "age": int(prow.age), "gender": prow.gender, "careunit": prow.careunit,
                "admit_reason": prow.admit_reason, "comorbidities": comorbidities,
                "has_cardiac_history": bool(prow.has_cardiac_history),
            },
            "complexity_context": {
                "hospital_complexity_tier": prow.hospital_complexity_tier,
                "entanglement_tier": prow.entanglement_tier,
                "population_context": prow.population_context,
                "socioeconomic_context": prow.socioeconomic_context,
            },
            "evidence_streams": {
                "continuous_signal": {
                    "provenance": vitals_provenance,
                    "n_readings": n_readings,
                    "readings": vitals_readings,
                },
                "intermittent_measurements": {
                    "provenance": labs_provenance,
                    "n_draws": n_draws,
                    "draws": labs_draws,
                },
                "delayed_narrative": {
                    "provenance": notes_provenance,
                    "notes": note_objs,
                },
                "diagnostic_studies": {
                    "ecg": [
                        (lambda base, parsed: {
                            **base,
                            **({k: v for k, v in parsed.items() if k not in ("subject_id", "provenance")} if parsed else {}),
                            "provenance": parsed["provenance"] if parsed else CSV_PROVENANCE,
                        })(
                            {k: (v if not isinstance(v, pd.Timestamp) else _iso(v)) for k, v in
                             {**e._asdict(), "ecg_time": _iso(e.ecg_time), "report_time": _iso(e.report_time)}.items()
                             if k != "Index"},
                            muse_by_ecg_row.get(e.ecg_id),
                        )
                        for e in pe.itertuples()
                    ],
                    "echo": [
                        {**{k: (v if not isinstance(v, pd.Timestamp) else _iso(v)) for k, v in
                            {**e._asdict(), "echo_time": _iso(e.echo_time), "report_time": _iso(e.report_time)}.items()
                            if k != "Index"}, "provenance": CSV_PROVENANCE}
                        for e in pec.itertuples()
                    ],
                },
            },
        }
        documents[str(hadm_id)] = doc

    return documents


def render_markdown(doc):
    a, c, e = doc["admission"], doc["complexity_context"], doc["evidence_streams"]
    lines = []
    lines.append(f"# Admission {doc['hadm_id']}  (patient {doc['subject_id']})\n")
    lines.append(f"**{a['age']}yo {a['gender']}**, {a['careunit']}, admitted for *{a['admit_reason']}*")
    lines.append(f"{a['admittime']} \u2192 {a['dischtime']}\n")
    if a["comorbidities"]:
        lines.append(f"**Comorbidities:** {', '.join(a['comorbidities'])}\n")
    lines.append(f"**Design context:** {c['hospital_complexity_tier']} \u00b7 {c['entanglement_tier']} entanglement \u00b7 "
                  f"{c['population_context']} \u00b7 {c['socioeconomic_context']}\n")

    lines.append("## Narrative timeline\n")
    for n in e["delayed_narrative"]["notes"]:
        if n.get("is_amendment"):
            lines.append(f"**[AMENDMENT \u2192 corrects {n['amends_note_id']}]** ({n['author_type']}) \u2014 "
                          f"event: `{n['event_charttime']}`, charted: `{n['storetime']}` "
                          f"\u2014 *{n.get('amendment_reason', '')}*")
        else:
            lines.append(f"**[{n['note_type']}]** ({n['author_type']}) \u2014 event: `{n['event_charttime']}`, "
                          f"charted: `{n['storetime']}`")
        lines.append(f"> {n['text']}")
        if n["medication_mentions"]:
            tags = ", ".join(f"{m['annotation']}=\u201c{m['text']}\u201d" for m in n["medication_mentions"])
            lines.append(f"medication spans: {tags}")
        lines.append("")

    if e["diagnostic_studies"]["ecg"]:
        lines.append("## ECG studies\n")
        for ecg_e in e["diagnostic_studies"]["ecg"]:
            lines.append(f"- {ecg_e['ecg_time']}: {ecg_e['finding']} (reported {ecg_e['report_time']})")
        lines.append("")

    if e["diagnostic_studies"]["echo"]:
        lines.append("## Echo studies\n")
        for echo_e in e["diagnostic_studies"]["echo"]:
            lines.append(f"- {echo_e['echo_time']}: {echo_e['finding']}, LVEF {echo_e['lvef_percent']}% "
                          f"(reported {echo_e['report_time']})")
        lines.append("")

    lines.append(f"## Summary counts\n")
    lines.append(f"- {e['continuous_signal']['n_readings']} continuous vitals readings")
    lines.append(f"- {e['intermittent_measurements']['n_draws']} lab draws")
    lines.append(f"- {len(e['delayed_narrative']['notes'])} notes")

    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    import jsonschema

    parser = argparse.ArgumentParser(description="Rebuild synthetic_documents.json from current CSVs + raw files.")
    parser.add_argument("--data-dir", default=".",
                         help="Folder containing the CSVs, document_schema.json, and XML_JSON/. "
                              "Run with --data-dir ./data if you're running this from the Tutorial/ root "
                              "rather than from inside Tutorial/data/.")
    args = parser.parse_args()
    data_dir = args.data_dir

    docs = build_documents(data_dir)
    print("assembled", len(docs), "documents")

    with open(f"{data_dir}/synthetic_documents.json", "w") as f:
        json.dump(docs, f, indent=2)

    with open(f"{data_dir}/document_schema.json") as f:
        schema = json.load(f)

    n_checked = 0
    for hadm_id, doc in docs.items():
        jsonschema.validate(instance=doc, schema=schema)
        n_checked += 1
    print(f"validated {n_checked}/{len(docs)} documents against document_schema.json: all passed")

    # render one document with medication mentions as the example
    example = next(d for d in docs.values() if any(
        n["medication_mentions"] for n in d["evidence_streams"]["delayed_narrative"]["notes"]
    ))
    md = render_markdown(example)
    out_name = f"{data_dir}/example_chart_{example['hadm_id']}.md"
    with open(out_name, "w") as f:
        f.write(md)
    print("wrote", out_name)
