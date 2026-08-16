"""
generate_amendments.py

Augments synthetic_notes.csv + synthetic_medication_labels.csv with a
document-versioning layer this corpus has never had: some medication-bearing
notes get a later correction, an amendment, the kind of thing that happens
constantly in real EHRs. Until now every note in this corpus was final the
moment it was charted. This makes some of them not final.

Run once, after synthetic_generator.py, before document_representation.py's
build_documents():

    python generate_amendments.py --data-dir .

DESIGN DECISION, worth understanding before you read the code: an
amendment's event_charttime is copied from the note it corrects, not set to
"now". The clinical event didn't move, only the documentation of it did.
Run it through plot_patient_timeline() or plot_lag_by_tier() with no other
changes and an amendment shows up as a note with an enormous lag, for free,
because that IS what it is. The record didn't just arrive late once, it
arrived, was believed, and then had to be walked back.

Patients with a real FHIR bundle on disk (XML_JSON/fhir_export/*.json) are
automatically excluded as amendment candidates: build_documents() takes the
FHIR branch exclusively for those patients, whose note_ids come from the
bundle itself, not this corpus's "{subject_id}-N{n}" scheme, so an
amendment referencing a CSV note_id would be a dangling reference for them.

Follows the corpus's own conventions exactly, matching synthetic_generator.py:
note_id = "{orig.note_id}-AMEND1" (never collides
with the "{subject_id}-N{n}" or "{subject_id}-N9xxxx" schemes already in
use), author_type "clinician" (corrections are clinician-entered),
medication swaps drawn from the same MEDICATIONS formulary so a corrected
drug is always a real, valid formulary entry, and Group codes following the
same "1_{last6}" pattern used by add_medication_mentions().

New columns on synthetic_notes.csv (added if not already present):
  is_amendment      1 for amendment rows, 0 otherwise
  amends_note_id    the note_id being corrected (NaN for non-amendments)
  amendment_reason  why (NaN for non-amendments)

New rows on synthetic_medication_labels.csv, same 7-column schema as
existing rows, pointing at the amendment's own note_id, so the corrected
value is exactly as recoverable via text[start:end] as the original was.

WARNING FOR BLOCK 2: alignment.naive_note_signal() counts notes and checks
for concern flags directly off synthetic_notes.csv. This script's rows use
note_type="amendment", which alignment.py now explicitly excludes (see the
one-line filter at the top of naive_note_signal, added alongside this
script) specifically so Block 2's numbers don't shift silently just because
Block 1 grew a new capability. If you ever regenerate alignment.py from
scratch, keep that filter.
"""

import argparse
import glob
import re
import pandas as pd
import numpy as np
from datetime import datetime

# Inlined from synthetic_generator.py rather than imported: generate_amendments.py
# and synthetic_generator.py don't always live in the same folder (the
# generator tends to sit alongside the CSVs it produces, this script sits
# alongside the other tutorial modules), so importing across that boundary
# is one path assumption too many. Keep MEDICATIONS in sync with
# synthetic_generator.py's own copy by hand if you ever add a drug there.
MEDICATIONS = [
    dict(name="metoprolol", brand="Lopressor", dose="25mg", mode="PO", freq="BID", reasons=["rate control", "hypertension"]),
    dict(name="furosemide", brand="Lasix", dose="40mg", mode="IV", freq="daily", reasons=["volume overload", "heart failure"]),
    dict(name="heparin", brand=None, dose="5000 units", mode="subcutaneous", freq="every 8 hours", reasons=["DVT prophylaxis"]),
    dict(name="insulin glargine", brand=None, dose="10 units", mode="subcutaneous", freq="nightly", reasons=["glycemic control"]),
    dict(name="levofloxacin", brand=None, dose="500mg", mode="IV", freq="daily", reasons=["pneumonia", "presumed infection"]),
    dict(name="albuterol", brand=None, dose="2.5mg", mode="nebulized", freq="every 4 hours as needed", reasons=["wheezing", "dyspnea"]),
    dict(name="apixaban", brand="Eliquis", dose="5mg", mode="PO", freq="BID", reasons=["atrial fibrillation", "anticoagulation"]),
    dict(name="lisinopril", brand="Prinivil", dose="10mg", mode="PO", freq="daily", reasons=["hypertension", "afterload reduction"]),
]


def round_datetimes(df, freq="min"):
    """Round every datetime-like column to the nearest minute (real hospital
    charting precision), same helper as synthetic_generator.py's own, copied
    rather than imported for the same reason MEDICATIONS is above."""
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.round(freq)
        elif df[col].dtype == object:
            sample = df[col].dropna()
            if len(sample) and isinstance(sample.iloc[0], datetime):
                df[col] = pd.to_datetime(df[col]).dt.round(freq)
    return df


REASONS = {"DOSAGE": "dosage transcription error", "MEDICATION": "wrong drug charted, corrected to actual order"}


def _fhir_covered_subjects(data_dir):
    """Subject ids with a real FHIR bundle on disk, under data_dir/XML_JSON/fhir_export/
    (matching document_representation.py's own discovery path). build_documents()
    takes the FHIR branch exclusively for these patients, and FHIR-sourced
    note_ids come from the bundle's own DocumentReference.id, a completely
    different scheme from this corpus's "{subject_id}-N{n}". An amendment
    generated against a CSV note_id would be a dangling reference for these
    patients, so they're excluded as candidates rather than risk producing an
    amends_note_id that points at nothing."""
    ids = set()
    for path in glob.glob(f"{data_dir}/XML_JSON/fhir_export/*.json"):
        m = re.search(r"(\d+)\.json$", path)
        if m:
            ids.add(int(m.group(1)))
    return ids


def _fmt_num(x):
    return str(int(x)) if float(x).is_integer() else f"{x:g}"


def _perturb_dose(dose_text, rng):
    """'500mg' -> '1000mg' or '250mg', '10 units' -> '20 units' or '5 units'.
    Doubling/halving mirrors the two most common real transcription errors
    (a missed or duplicated leading digit). Returns None for anything that
    doesn't parse as <number><unit>, so the caller can fall back cleanly."""
    m = re.match(r"^([\d.]+)\s*(.*)$", dose_text.strip())
    if not m:
        return None
    value = float(m.group(1))
    factor = rng.choice([0.5, 2.0])
    new_value = value * factor
    return f"{_fmt_num(new_value)}{m.group(2)}"


def generate_amendments(data_dir=".", amend_rate=0.15, seed=7,
                         notes_path=None, meds_path=None, out_notes_path=None, out_meds_path=None):
    """Returns (notes_df, meds_df, n_amendments_added). Writes CSVs in place
    unless out_notes_path=False / out_meds_path=False (DataFrames only)."""
    rng = np.random.default_rng(seed)

    notes_path = notes_path or f"{data_dir}/synthetic_notes.csv"
    meds_path = meds_path or f"{data_dir}/synthetic_medication_labels.csv"
    notes = pd.read_csv(notes_path, parse_dates=["event_charttime", "storetime"])
    meds = pd.read_csv(meds_path)

    for col, default in [("is_amendment", 0), ("amends_note_id", np.nan), ("amendment_reason", np.nan)]:
        if col not in notes.columns:
            notes[col] = default
    notes["is_amendment"] = notes["is_amendment"].fillna(0).astype(int)

    already_amended = set(notes.loc[notes.is_amendment == 1, "amends_note_id"].dropna())
    fhir_covered = _fhir_covered_subjects(data_dir)
    med_notes = meds[meds.Annotation.isin(["DOSAGE", "MEDICATION"])].note_id.unique()
    candidates = notes[
        notes.note_id.isin(med_notes) & (notes.is_amendment == 0) & (~notes.note_id.isin(already_amended))
        & (~notes.subject_id.isin(fhir_covered))
    ]
    n_amend = min(len(candidates), max(1, round(len(candidates) * amend_rate))) if len(candidates) else 0
    chosen = candidates.sample(n=n_amend, random_state=seed) if n_amend else candidates.iloc[0:0]

    new_note_rows, new_med_rows = [], []
    for _, orig in chosen.iterrows():
        orig_meds = meds[(meds.note_id == orig.note_id) & (meds.Annotation.isin(["DOSAGE", "MEDICATION"]))]
        if not len(orig_meds):
            continue
        target = orig_meds.sample(n=1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]

        if target.Annotation == "DOSAGE":
            corrected_value = _perturb_dose(target.Text, rng)
            if corrected_value is None:
                continue
        else:  # MEDICATION: swap to a different real formulary drug entirely
            other = [m for m in MEDICATIONS if m["name"] != target.Text and m.get("brand") != target.Text]
            if not other:
                continue
            corrected_value = other[int(rng.integers(0, len(other)))]["name"]
        reason = REASONS[target.Annotation]

        amend_note_id = f"{orig.note_id}-AMEND1"
        amend_storetime = orig.storetime + pd.Timedelta(hours=float(rng.uniform(2, 30)))
        amend_text = (
            f"ADDENDUM to note {orig.note_id}: {target.Annotation.lower()} was charted incorrectly. "
            f"Corrected {target.Annotation.lower()}: {corrected_value} (previously {target.Text})."
        )
        start = amend_text.index(corrected_value)
        end = start + len(corrected_value)
        group_code = f"1_{amend_note_id.replace('-', '')[-6:]}"

        new_note_rows.append(dict(
            note_id=amend_note_id, subject_id=orig.subject_id, hadm_id=orig.hadm_id,
            note_type="amendment", author_type="clinician",
            event_charttime=orig.event_charttime,   # the event didn't move, the documentation of it did
            storetime=amend_storetime, text=amend_text,
            history_reliability=np.nan, related_event_id=np.nan,
            is_amendment=1, amends_note_id=orig.note_id, amendment_reason=reason,
        ))
        new_med_rows.append(dict(
            subject_id=orig.subject_id, note_id=amend_note_id,
            **{"Start Position": start, "End Position": end},
            Annotation=target.Annotation, Group=group_code, Text=corrected_value,
        ))

    notes_out = pd.concat([notes, pd.DataFrame(new_note_rows)], ignore_index=True) if new_note_rows else notes
    notes_out = round_datetimes(notes_out).sort_values(["subject_id", "storetime"]).reset_index(drop=True)
    meds_out = pd.concat([meds, pd.DataFrame(new_med_rows)], ignore_index=True) if new_med_rows else meds

    if out_notes_path is not False:
        notes_out.to_csv(out_notes_path or notes_path, index=False)
    if out_meds_path is not False:
        meds_out.to_csv(out_meds_path or meds_path, index=False)

    return notes_out, meds_out, len(new_note_rows)


def resolve_as_of(notes, meds, as_of_time=None):
    """What the record actually said at `as_of_time` (None = full current
    state, amendments included). Notes not yet charted by as_of_time are
    excluded outright; amendments charted after as_of_time don't count yet,
    so the note they target correctly still shows its original,
    not-yet-corrected value."""
    if as_of_time is None:
        return notes, meds
    visible_notes = notes[notes.storetime <= pd.Timestamp(as_of_time)].copy()
    visible_meds = meds[meds.note_id.isin(visible_notes.note_id)].copy()
    return visible_notes, visible_meds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--amend-rate", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    _, _, n = generate_amendments(data_dir=args.data_dir, amend_rate=args.amend_rate, seed=args.seed)
    print(f"Added {n} amendment note(s). synthetic_notes.csv and synthetic_medication_labels.csv updated in place.")
