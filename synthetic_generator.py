"""
synthetic_generator.py

Generates synthetic ICU-like patient data for the DocEng '26 tutorial,
"Temporally Entangled Documents: Multimodal AI for ICU Records Under Label Ambiguity".

No real patient data is used or reproduced anywhere in this script. Timing
parameters (documentation lag, lab result lag, vitals sampling interval,
length of stay) are calibrated against summary statistics computed from the
uploaded MIMIC-derived sample files, not against any individual real record.

Two independent complexity axes, plus one compounding population axis:
  - hospital_complexity_tier : how much data a stay generates (A/B/C)
  - entanglement_tier        : how asynchronous/lagged/uncertain it is (low/medium/high)
  - population_context       : neurodivergent / disability / dementia (mild/moderate/severe) / none
  - socioeconomic_context    : under_resourced flag (independent, compounding)

Design principle: raw files contain only what a real hospital system would
actually give you, timestamps and observations. Nothing that requires
comparing two timestamps (documentation lag, time-to-result, etc.) is
pre-computed anywhere in the output. That comparison is left as a Block 1
exercise for tutorial participants. The one exception is `synthetic_labels.csv`,
which plays the role of the instructor's answer key: it carries the
generator's ground truth (true onset time, the uncertainty band around it),
values that are not derivable from the raw tables at all, since no hospital
system records "true onset of clinical deterioration" directly. That is a
different kind of thing from a lag, which the raw tables already contain the
ingredients for.

Output: eight CSVs written to OUT_DIR.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

rng = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_PATIENTS = 75
BASE_SUBJECT_ID = 90000001
BASE_HADM_ID = 92000001
BASE_STAY_ID = 93000001

OUT_DIR = "."  # overridden by --data-dir when run as a script, see __main__ below

CAREUNITS = ["MICU", "SICU", "CCU", "CVICU"]

HOSPITAL_TIERS = ["A_minimal", "B_moderate", "C_complex"]
ENTANGLEMENT_TIERS = ["low", "medium", "high"]
POPULATION_CONTEXTS = [
    "none", "neurodivergent", "disability",
    "dementia_mild", "dementia_moderate", "dementia_severe",
]
POP_PROBS = [0.45, 0.15, 0.15, 0.10, 0.08, 0.07]

# Entanglement tier base parameters, calibrated loosely against real stats:
#   radiology lag median 2h10m / 90th pct 21h35m
#   lab lag median 74 min
#   ED vitals sampling gap median 92 min
ENTANGLEMENT_PARAMS = {
    "low":    dict(note_lag_mult=1.0, onset_uncertainty_h=0.5,  lab_lag_mult=1.0, missing_base=0.02),
    "medium": dict(note_lag_mult=1.8, onset_uncertainty_h=2.0,  lab_lag_mult=1.6, missing_base=0.05),
    "high":   dict(note_lag_mult=3.0, onset_uncertainty_h=6.0,  lab_lag_mult=2.4, missing_base=0.12),
}

HOSPITAL_PARAMS = {
    # los_lognorm_mean (days), vitals_interval_min, labs_per_day, notes_count_range
    "A_minimal":  dict(los_mean=1.0, vitals_interval=60, labs_per_day=1.5, notes_range=(1, 2)),
    "B_moderate": dict(los_mean=2.2, vitals_interval=30, labs_per_day=2.5, notes_range=(2, 3)),
    "C_complex":  dict(los_mean=4.5, vitals_interval=15, labs_per_day=4.0, notes_range=(3, 5)),
}

LABS = {
    "Lactate":    dict(unit="mmol/L", normal=(0.5, 2.0), abnormal=(3.0, 8.0)),
    "Potassium":  dict(unit="mEq/L",  normal=(3.5, 5.0), abnormal=(5.5, 6.5)),
    "Creatinine": dict(unit="mg/dL",  normal=(0.6, 1.3), abnormal=(1.8, 3.5)),
    "WBC":        dict(unit="K/uL",   normal=(4.5, 11.0), abnormal=(13.0, 22.0)),
    "Troponin":   dict(unit="ng/mL",  normal=(0.0, 0.04), abnormal=(0.5, 5.0)),
}

NOTE_TEMPLATES = {
    "admission": [
        "Patient admitted for {reason}. Baseline vitals reviewed. Plan: monitor and treat per protocol.",
        "Admitted from {source} with {reason}. Initial assessment stable. Continuing workup.",
    ],
    "routine_stable": [
        "No acute changes overnight. Patient stable, tolerating current plan of care.",
        "Overnight course unremarkable. Vitals within patient's usual range. Continue monitoring.",
    ],
    "event_clinician": [
        "Patient noted to be less responsive than earlier today, per nursing report. Vitals reviewed, mild instability noted. Continuing to monitor closely.",
        "Change in patient status observed on rounds. Trending vitals reviewed. Plan updated, further workup pending.",
        "Nursing reports patient appears more unwell since last check. Reassessed, orders updated accordingly.",
    ],
    "event_overshadowed": [
        "Change in presentation noted, consistent with patient's known baseline condition. No acute intervention indicated at this time; continue current plan.",
        "Behaviour and status change observed, attributed to patient's existing diagnosis. Will continue to monitor as per baseline pattern.",
    ],
    "caregiver": [
        "Family visited and reports patient seems different from their usual baseline today.",
        "Caregiver present at bedside, notes patient is more unsettled than is typical for them.",
        "Group home / family contact reports a change in patient's usual presentation over the past day.",
    ],
    "discharge": [
        "Patient's condition improved / stabilized over the admission. Discharged with follow-up plan in place.",
    ],
}

ADMIT_REASONS = [
    "shortness of breath", "chest pain", "altered mental status", "sepsis workup",
    "post-procedural monitoring", "acute kidney injury", "arrhythmia", "hypotension",
]

CARDIAC_COMORBIDITIES = [
    "hypertension", "prior myocardial infarction", "heart failure",
    "atrial fibrillation", "coronary artery disease",
]
OTHER_COMORBIDITIES = ["type 2 diabetes", "COPD", "chronic kidney disease"]

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
BRAND_NAME_PROB = 0.40  # how often a mention uses the brand name instead of generic, for meds that have one

# Several phrasings per clinical moment, so the same medication doesn't always
# read the same way, and different note types get language that actually fits
# what a clinician would write at that point in the stay. Each template is a
# sequence of (piece, annotation) tuples; piece is either a literal string
# (annotation=None) or a key into the medication/reason values (annotation set).
MEDICATION_TEMPLATES = {
    "start": [
        [("Started ", None), ("name", "MEDICATION"), (" ", None), ("dose", "DOSAGE"), (" ", None),
         ("mode", "MODE"), (" ", None), ("freq", "FREQUENCY"), (" for ", None), ("reason", "REASON"), (".", None)],
        [("Plan: start ", None), ("name", "MEDICATION"), (" ", None), ("dose", "DOSAGE"), (", ", None),
         ("mode", "MODE"), (", ", None), ("freq", "FREQUENCY"), (" for ", None), ("reason", "REASON"), (".", None)],
    ],
    "continue": [
        [("Continues on ", None), ("name", "MEDICATION"), (" ", None), ("dose", "DOSAGE"), (" ", None),
         ("mode", "MODE"), (" ", None), ("freq", "FREQUENCY"), (" for ", None), ("reason", "REASON"),
         (", tolerating well.", None)],
        [("Home medication ", None), ("name", "MEDICATION"), (" (", None), ("dose", "DOSAGE"), (", ", None),
         ("mode", "MODE"), (", ", None), ("freq", "FREQUENCY"), (") continued for ", None),
         ("reason", "REASON"), (".", None)],
    ],
    "discharge": [
        [("Discharge medications include ", None), ("name", "MEDICATION"), (" ", None), ("dose", "DOSAGE"),
         (" ", None), ("mode", "MODE"), (" ", None), ("freq", "FREQUENCY"), (" for ", None),
         ("reason", "REASON"), (".", None)],
    ],
}
NOTE_TYPE_TO_TEMPLATE_CATEGORY = {"admission": "start", "routine_stable": "continue", "discharge": "discharge"}
NOTE_TYPE_MENTION_BASE_PROB = {"admission": 0.35, "routine_stable": 0.22, "discharge": 0.40}

# Extra narrative sentences with no medication content at all, surrounding the
# medication sentence so it's genuinely embedded in a paragraph rather than
# being most of the note. This is what makes span extraction a real search
# problem instead of "the note basically is the answer."
FILLER_SENTENCES = {
    "start": [
        "Vitals reviewed on arrival, patient alert and oriented.",
        "No acute distress noted at this time.",
        "Case discussed with the primary team.",
        "Further workup pending, will reassess in the morning.",
        "Allergies reviewed, none reported.",
        "Baseline labs sent, results pending.",
        "Patient and family updated on initial plan.",
        "Code status confirmed on admission.",
    ],
    "continue": [
        "Overnight vitals within the patient's acceptable range.",
        "Family updated on plan of care at bedside.",
        "No new complaints reported by patient this shift.",
        "Mobility and appetite both improving steadily.",
        "Pain well controlled on current regimen.",
        "Physical therapy assessment completed today.",
        "Input and output balanced over the last shift.",
        "Sleeping well overnight per patient report.",
    ],
    "discharge": [
        "Patient ambulating independently prior to discharge.",
        "Follow-up appointment scheduled with primary care.",
        "Discharge instructions reviewed with patient and family.",
        "Patient verbalizes understanding of the discharge plan.",
        "No acute issues at time of discharge.",
        "Transportation home arranged with family.",
        "Wound care instructions provided, if applicable.",
        "Return precautions reviewed with patient.",
    ],
}

INTERPRETATION_TEMPLATES = {
    "ecg_interpretation": [
        "ECG reviewed: {finding}. Clinically correlated with current presentation, plan unchanged.",
        "Reviewed today's ECG, {finding} noted. Will trend with serial tracings.",
        "ECG findings ({finding}) discussed on rounds, no acute intervention needed at this time.",
    ],
    "echo_interpretation": [
        "Echo reviewed: {finding} LVEF {lvef}%. Findings incorporated into management plan.",
        "Reviewed echocardiogram, {finding} LVEF {lvef}%. Cardiology aware.",
    ],
}

def round_datetimes(df, freq="min"):
    """Round every datetime-like column to the nearest minute (real hospital
    charting precision) before writing to CSV. Raw Python datetime arithmetic
    with float hours/days leaves microsecond precision, which spreadsheet
    apps reliably mis-parse as a time fragment rather than a full timestamp."""
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.round(freq)
        elif df[col].dtype == object:
            # columns of raw python datetime objects (not yet cast to datetime64)
            sample = df[col].dropna()
            if len(sample) and isinstance(sample.iloc[0], datetime):
                df[col] = pd.to_datetime(df[col]).dt.round(freq)
    return df


# ---------------------------------------------------------------------------
# 1. Patients
# ---------------------------------------------------------------------------

def generate_patients():
    rows = []
    admit_window_start = datetime(2024, 1, 1)
    for i in range(N_PATIENTS):
        subject_id = BASE_SUBJECT_ID + i
        hadm_id = BASE_HADM_ID + i
        stay_id = BASE_STAY_ID + i

        hosp_tier = rng.choice(HOSPITAL_TIERS)
        ent_tier = rng.choice(ENTANGLEMENT_TIERS)
        pop_ctx = rng.choice(POPULATION_CONTEXTS, p=POP_PROBS)
        under_resourced = bool(rng.random() < 0.30)
        outcome = int(rng.random() < 0.55)

        age = int(np.clip(rng.normal(65, 15), 19, 95))
        gender = rng.choice(["F", "M"])
        careunit = rng.choice(CAREUNITS)

        # comorbidities: independent of admit_reason, drives ECG/echo ordering probability
        n_cardiac = rng.choice([0, 1, 2], p=[0.55, 0.32, 0.13])
        n_other = rng.choice([0, 1], p=[0.65, 0.35])
        comorbid_list = list(rng.choice(CARDIAC_COMORBIDITIES, size=n_cardiac, replace=False)) if n_cardiac else []
        comorbid_list += list(rng.choice(OTHER_COMORBIDITIES, size=n_other, replace=False)) if n_other else []
        comorbidities = ", ".join(comorbid_list) if comorbid_list else "none"
        has_cardiac_history = len(comorbid_list) > 0 and any(c in CARDIAC_COMORBIDITIES for c in comorbid_list)

        hp = HOSPITAL_PARAMS[hosp_tier]
        los_days = float(np.clip(rng.lognormal(mean=np.log(hp["los_mean"]), sigma=0.45), 0.3, 14.0))

        admittime = admit_window_start + timedelta(
            days=float(rng.uniform(0, 700)), hours=float(rng.uniform(0, 23))
        )
        dischtime = admittime + timedelta(days=los_days)

        true_onset_time = None
        if outcome == 1:
            # onset somewhere between 20% and 80% of the stay
            frac = rng.uniform(0.2, 0.8)
            true_onset_time = admittime + timedelta(days=los_days * frac)

        rows.append(dict(
            subject_id=subject_id, hadm_id=hadm_id, stay_id=stay_id,
            age=age, gender=gender, careunit=careunit,
            admittime=admittime, dischtime=dischtime,
            los_days=round(los_days, 3),
            hospital_complexity_tier=hosp_tier,
            entanglement_tier=ent_tier,
            population_context=pop_ctx,
            socioeconomic_context="under_resourced" if under_resourced else "standard_resourced",
            outcome=outcome,
            true_onset_time=true_onset_time,
            admit_reason=rng.choice(ADMIT_REASONS),
            comorbidities=comorbidities,
            has_cardiac_history=has_cardiac_history,
        ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Modifier helpers (population + socioeconomic compounding on top of entanglement tier)
# ---------------------------------------------------------------------------

def dementia_severity(pop_ctx):
    return {"dementia_mild": 1, "dementia_moderate": 2, "dementia_severe": 3}.get(pop_ctx, 0)


def onset_uncertainty_hours(row):
    base = ENTANGLEMENT_PARAMS[row.entanglement_tier]["onset_uncertainty_h"]
    sev = dementia_severity(row.population_context)
    mult = {0: 1.0, 1: 1.3, 2: 1.8, 3: 2.5}[sev]
    return base * mult


def note_lag_multiplier(row):
    mult = ENTANGLEMENT_PARAMS[row.entanglement_tier]["note_lag_mult"]
    sev = dementia_severity(row.population_context)
    mult *= {0: 1.0, 1: 1.2, 2: 1.4, 3: 1.7}[sev]
    if row.population_context == "disability":
        mult *= 1.3
    if row.socioeconomic_context == "under_resourced":
        mult *= 1.3
    return mult


def vitals_missing_rate(row):
    rate = ENTANGLEMENT_PARAMS[row.entanglement_tier]["missing_base"]
    if row.population_context == "disability":
        rate += 0.15
    if row.socioeconomic_context == "under_resourced":
        rate += 0.10
    return min(rate, 0.5)


def vitals_noise_multiplier(row):
    return 1.6 if row.population_context == "neurodivergent" else 1.0


def lab_lag_multiplier(row):
    mult = ENTANGLEMENT_PARAMS[row.entanglement_tier]["lab_lag_mult"]
    if row.socioeconomic_context == "under_resourced":
        mult *= 1.3
    return mult


# ---------------------------------------------------------------------------
# 2. Continuous vitals stream
# ---------------------------------------------------------------------------

BASELINE_VITALS = dict(heartrate=80, resprate=18, o2sat=97, sbp=120, dbp=75)
BASELINE_SD = dict(heartrate=6, resprate=2.5, o2sat=1.2, sbp=10, dbp=7)
DETERIORATION_SHIFT = dict(heartrate=+38, resprate=+10, o2sat=-11, sbp=-28, dbp=-14)


def generate_vitals(patients):
    rows = []
    for row in patients.itertuples():
        hp = HOSPITAL_PARAMS[row.hospital_complexity_tier]
        interval_min = hp["vitals_interval"]
        missing_rate = vitals_missing_rate(row)
        noise_mult = vitals_noise_multiplier(row)
        # individual baseline offset (patient-level variation)
        indiv_offset = {k: rng.normal(0, BASELINE_SD[k] * 0.4) for k in BASELINE_VITALS}

        n_readings = int((row.los_days * 24 * 60) / interval_min)
        t = row.admittime
        ramp_hours = 3.0  # deterioration develops over ~3h once onset begins
        for _ in range(n_readings):
            t = t + timedelta(minutes=interval_min)
            if t >= row.dischtime:
                break

            # deterioration progress in [0, 1]
            prog = 0.0
            if row.outcome == 1 and pd.notna(row.true_onset_time) and t >= row.true_onset_time:
                hrs_since = (t - row.true_onset_time).total_seconds() / 3600
                prog = min(1.0, hrs_since / ramp_hours)

            reading = {}
            for k, base in BASELINE_VITALS.items():
                val = (base + indiv_offset[k]
                       + prog * DETERIORATION_SHIFT[k]
                       + rng.normal(0, BASELINE_SD[k] * noise_mult))
                reading[k] = round(val, 1)

            missing = rng.random() < missing_rate
            rows.append(dict(
                subject_id=row.subject_id, stay_id=row.stay_id, charttime=t,
                heartrate=None if missing else reading["heartrate"],
                resprate=None if missing else reading["resprate"],
                o2sat=None if missing else reading["o2sat"],
                sbp=None if missing else reading["sbp"],
                dbp=None if missing else reading["dbp"],
                signal_quality="degraded" if missing else "normal",
            ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Intermittent labs stream
# ---------------------------------------------------------------------------

def generate_labs(patients):
    rows = []
    specimen_counter = 1
    for row in patients.itertuples():
        hp = HOSPITAL_PARAMS[row.hospital_complexity_tier]
        n_draws = max(1, int(row.los_days * hp["labs_per_day"]))
        lag_mult = lab_lag_multiplier(row)
        missing_rate = vitals_missing_rate(row) * 0.6  # labs slightly more robustly captured than vitals

        draw_times = sorted(
            row.admittime + timedelta(hours=float(h))
            for h in rng.uniform(0, row.los_days * 24, size=n_draws)
        )
        for dt in draw_times:
            specimen_id = specimen_counter
            specimen_counter += 1
            panel = rng.choice(list(LABS.keys()), size=min(4, len(LABS)), replace=False)

            prog = 0.0
            if row.outcome == 1 and pd.notna(row.true_onset_time) and dt >= row.true_onset_time:
                hrs_since = (dt - row.true_onset_time).total_seconds() / 3600
                prog = min(1.0, hrs_since / 6.0)

            # lag: charttime (drawn) -> storetime (resulted), lognormal calibrated ~74 min median
            lag_minutes = float(rng.lognormal(mean=np.log(74), sigma=0.5)) * lag_mult
            storetime = dt + timedelta(minutes=lag_minutes)

            for lab_name in panel:
                spec = LABS[lab_name]
                lo, hi = spec["normal"]
                alo, ahi = spec["abnormal"]
                normal_val = rng.uniform(lo, hi)
                abnormal_val = rng.uniform(alo, ahi)
                value = normal_val + prog * (abnormal_val - normal_val)
                value = round(float(value), 2)
                flag = "abnormal" if (value < lo or value > hi) else "normal"

                missing_value = rng.random() < missing_rate
                rows.append(dict(
                    subject_id=row.subject_id, hadm_id=row.hadm_id, specimen_id=specimen_id,
                    charttime=dt, storetime=storetime,
                    label=lab_name, value=None if missing_value else value,
                    valueuom=spec["unit"], flag=None if missing_value else flag,
                ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Notes stream (clinician + caregiver, unified with an author_type column)
# ---------------------------------------------------------------------------

def generate_notes(patients):
    rows = []
    note_counter = 1
    for row in patients.itertuples():
        hp = HOSPITAL_PARAMS[row.hospital_complexity_tier]
        n_notes_base = int(rng.integers(hp["notes_range"][0], hp["notes_range"][1] + 1))
        lag_mult = note_lag_multiplier(row)
        sev = dementia_severity(row.population_context)

        # -- admission note --
        adm_lag_h = float(rng.lognormal(mean=np.log(2.0), sigma=0.4)) * lag_mult
        rows.append(_make_note(
            note_counter, row, note_type="admission", author_type="clinician",
            event_charttime=row.admittime, storetime=row.admittime + timedelta(hours=adm_lag_h),
            text=rng.choice(NOTE_TEMPLATES["admission"]).format(
                reason=row.admit_reason, source="ED"),
            history_reliability="reliable" if sev == 0 else ("limited" if sev >= 2 else "mostly reliable"),
        ))
        note_counter += 1

        # -- routine notes spread through the stay (before/unrelated to onset) --
        n_routine = max(0, n_notes_base - 1)
        for _ in range(n_routine):
            t_ref = row.admittime + timedelta(hours=float(rng.uniform(0.5, max(1.0, row.los_days * 24 - 1))))
            lag_h = float(rng.lognormal(mean=np.log(3.0), sigma=0.5)) * lag_mult
            rows.append(_make_note(
                note_counter, row, note_type="routine_stable", author_type="clinician",
                event_charttime=t_ref, storetime=t_ref + timedelta(hours=lag_h),
                text=rng.choice(NOTE_TEMPLATES["routine_stable"]),
                history_reliability="reliable" if sev == 0 else ("limited" if sev >= 2 else "mostly reliable"),
            ))
            note_counter += 1

        # -- event-describing note, if outcome occurred --
        if row.outcome == 1 and pd.notna(row.true_onset_time):
            # overshadowing risk: disability or dementia -> some chance the change gets
            # attributed to baseline condition rather than investigated fresh
            overshadow_risk = row.population_context in (
                "disability", "dementia_mild", "dementia_moderate", "dementia_severe"
            )
            use_overshadow = overshadow_risk and rng.random() < 0.55
            template_pool = NOTE_TEMPLATES["event_overshadowed"] if use_overshadow else NOTE_TEMPLATES["event_clinician"]

            base_delay_h = float(rng.lognormal(mean=np.log(1.5), sigma=0.4))
            event_note_charttime = row.true_onset_time + timedelta(hours=base_delay_h)
            lag_h = float(rng.lognormal(mean=np.log(2.5), sigma=0.5)) * lag_mult
            if use_overshadow:
                lag_h *= 1.4  # overshadowing stretches lag further still
            rows.append(_make_note(
                note_counter, row, note_type="event_overshadowed" if use_overshadow else "event_clinician",
                author_type="clinician",
                event_charttime=event_note_charttime, storetime=event_note_charttime + timedelta(hours=lag_h),
                text=rng.choice(template_pool),
                history_reliability="reliable" if sev == 0 else ("limited" if sev >= 2 else "mostly reliable"),
            ))
            note_counter += 1

            # -- caregiver note: guaranteed for moderate/severe dementia, probabilistic otherwise --
            caregiver_prob = {0: 0.10, 1: 0.35, 2: 0.75, 3: 0.95}[sev]
            if row.population_context == "disability":
                caregiver_prob = max(caregiver_prob, 0.20)
            if rng.random() < caregiver_prob:
                cg_offset_h = float(rng.uniform(-4, 10))  # arrives irregularly, before or after the clinician note
                cg_time = event_note_charttime + timedelta(hours=cg_offset_h)
                cg_lag_h = float(rng.uniform(0.1, 3.0))  # informal, often charted close to when reported
                rows.append(_make_note(
                    note_counter, row, note_type="caregiver", author_type="caregiver",
                    event_charttime=cg_time, storetime=cg_time + timedelta(hours=cg_lag_h),
                    text=rng.choice(NOTE_TEMPLATES["caregiver"]),
                    history_reliability="n/a",
                ))
                note_counter += 1

        # -- discharge note --
        disc_lag_h = float(rng.lognormal(mean=np.log(4.0), sigma=0.5)) * lag_mult
        rows.append(_make_note(
            note_counter, row, note_type="discharge", author_type="clinician",
            event_charttime=row.dischtime, storetime=row.dischtime + timedelta(hours=disc_lag_h),
            text=rng.choice(NOTE_TEMPLATES["discharge"]),
            history_reliability="reliable" if sev == 0 else ("limited" if sev >= 2 else "mostly reliable"),
        ))
        note_counter += 1

    return pd.DataFrame(rows)


def _make_note(note_id, row, note_type, author_type, event_charttime, storetime, text, history_reliability):
    # Deliberately no precomputed lag column here: event_charttime and storetime
    # are both raw fields a real system would give you. Computing the gap
    # between them is the Block 1 exercise, not something to hand over.
    return dict(
        note_id=f"{row.subject_id}-N{note_id}", subject_id=row.subject_id, hadm_id=row.hadm_id,
        note_type=note_type, author_type=author_type,
        event_charttime=event_charttime, storetime=storetime,
        text=text, history_reliability=history_reliability,
    )


# ---------------------------------------------------------------------------
# 4b. Medication mentions with exact character-span labels, mirroring the
#     real medication-labels-mimic-note schema (Start Position, End Position,
#     Annotation, Group) exactly, so the same span-to-text join workflow
#     applies. We're generating the text ourselves, so spans are exact by
#     construction on this side, that's the participant's job to recover.
# ---------------------------------------------------------------------------

def add_medication_mentions(notes, patients):
    """Appends one medication sentence to a subset of admission/routine/discharge
    notes (weighted toward cardiac-history patients), drawing from several
    phrasings per clinical moment so the corpus doesn't read as one template
    repeated 41 times. Returns (updated_notes, medication_labels), where
    medication_labels carries the real dataset's four columns plus subject_id
    and the literal Text of each span, so the file is readable on its own."""
    notes = notes.copy()
    patients_by_id = patients.set_index("subject_id")
    label_rows = []

    eligible = notes[notes.note_type.isin(NOTE_TYPE_TO_TEMPLATE_CATEGORY.keys())]
    for idx in eligible.index:
        n = notes.loc[idx]
        prow = patients_by_id.loc[n.subject_id]
        base_p = NOTE_TYPE_MENTION_BASE_PROB[n.note_type]
        p_mention = base_p + (0.30 if prow.has_cardiac_history else 0)
        if rng.random() >= min(p_mention, 0.85):
            continue

        med = MEDICATIONS[rng.integers(0, len(MEDICATIONS))]
        reason = med["reasons"][rng.integers(0, len(med["reasons"]))]
        category = NOTE_TYPE_TO_TEMPLATE_CATEGORY[n.note_type]
        template = MEDICATION_TEMPLATES[category][rng.integers(0, len(MEDICATION_TEMPLATES[category]))]

        # real clinical documentation mixes brand and generic names freely;
        # a naive dictionary of generic names alone will miss these
        display_name = med["name"]
        if med["brand"] and rng.random() < BRAND_NAME_PROB:
            display_name = med["brand"]

        field_values = {"name": display_name, "dose": med["dose"], "mode": med["mode"],
                         "freq": med["freq"], "reason": reason}

        # surround the medication sentence with real narrative, so the span
        # is embedded in a paragraph rather than being most of the note
        filler_pool = FILLER_SENTENCES[category]
        n_before = int(rng.integers(1, 3))   # 1 or 2 filler sentences before
        n_after = int(rng.integers(1, 3))    # 1 or 2 filler sentences after
        fillers = rng.choice(filler_pool, size=min(n_before + n_after, len(filler_pool)), replace=False)
        n_before = min(n_before, len(fillers))
        before_text = " ".join(fillers[:n_before])
        after_text = " ".join(fillers[n_before:])

        base_text = n.text
        prefix = "" if base_text.endswith(" ") else " "
        lead_in = (before_text + " ") if before_text else ""
        cursor = len(base_text) + len(prefix) + len(lead_in)

        group_code = f"1_{n.note_id.replace('-', '')[-6:]}"
        sentence_chunks = []
        for key_or_text, annotation in template:
            piece = field_values[key_or_text] if annotation else key_or_text
            start = cursor
            end = start + len(piece)
            if annotation:
                label_rows.append(dict(subject_id=n.subject_id, note_id=n.note_id, start=start, end=end,
                                        annotation=annotation, group=group_code, text=piece))
            sentence_chunks.append(piece)
            cursor = end

        trail_out = (" " + after_text) if after_text else ""
        notes.at[idx, "text"] = base_text + prefix + lead_in + "".join(sentence_chunks) + trail_out

    med_labels = pd.DataFrame(label_rows)
    if len(med_labels):
        med_labels = med_labels.rename(columns={
            "start": "Start Position", "end": "End Position",
            "annotation": "Annotation", "group": "Group", "text": "Text",
        })[["subject_id", "note_id", "Start Position", "End Position", "Annotation", "Group", "Text"]]
    return notes, med_labels


# ---------------------------------------------------------------------------
# 5a. ECG events: structured interval/axis measurements, not raw waveform.
#     Mirrors Master_Sheet_Sample.csv's real schema (rr_interval, p/qrs/t
#     onset-end, axes) plus a short interpretive finding, the same shape as
#     a real ECG machine report without touching actual signal data.
# ---------------------------------------------------------------------------

ECG_FINDINGS_NORMAL = [
    "Normal sinus rhythm", "Sinus rhythm, otherwise normal ECG",
    "Sinus rhythm with occasional ectopy",
]
ECG_FINDINGS_ABNORMAL = [
    "Sinus tachycardia", "Atrial fibrillation", "Nonspecific ST-T wave abnormality",
    "Sinus bradycardia", "Prolonged QT interval", "Left ventricular hypertrophy",
]
CARDIAC_REASONS = {"arrhythmia", "chest pain", "hypotension"}


def _ecg_measurements(abnormal):
    rr = rng.normal(650 if not abnormal else 480, 60)
    return dict(
        rr_interval=round(rr, 0),
        p_onset=round(rng.normal(40, 6), 0), p_end=round(rng.normal(128, 8), 0),
        qrs_onset=round(rng.normal(165, 8), 0), qrs_end=round(rng.normal(248, 10), 0),
        t_end=round(rng.normal(480 if not abnormal else 520, 20), 0),
        p_axis=round(rng.normal(60, 15), 0),
        qrs_axis=round(rng.normal(45, 20) if not abnormal else rng.normal(20, 35), 0),
        t_axis=round(rng.normal(45, 15), 0),
    )


def generate_ecg_events(patients):
    rows = []
    ecg_counter = 1
    for row in patients.itertuples():
        p_ecg = (0.20 + (0.30 if row.admit_reason in CARDIAC_REASONS else 0)
                 + (0.35 if row.has_cardiac_history else 0)
                 + (0.15 if row.outcome == 1 else 0))
        if rng.random() >= min(p_ecg, 0.92):
            continue

        # baseline ECG near admission
        t0 = row.admittime + timedelta(hours=float(rng.uniform(0.5, 4)))
        report_lag_h = float(rng.lognormal(mean=np.log(0.5), sigma=0.5))
        abnormal0 = rng.random() < (0.15 if row.outcome == 0 else 0.35)
        meas = _ecg_measurements(abnormal0)
        rows.append(dict(
            ecg_id=f"{row.subject_id}-ECG{ecg_counter}", subject_id=row.subject_id, hadm_id=row.hadm_id,
            ecg_time=t0, report_time=t0 + timedelta(hours=report_lag_h),
            finding=rng.choice(ECG_FINDINGS_ABNORMAL) if abnormal0 else rng.choice(ECG_FINDINGS_NORMAL),
            **meas,
        ))
        ecg_counter += 1

        # repeat ECG around the event, if one occurred, more ECGs get ordered
        # when a patient deteriorates: a realistic density artefact, worth
        # letting participants discover in Block 2.
        if row.outcome == 1 and pd.notna(row.true_onset_time) and rng.random() < 0.55:
            t1 = row.true_onset_time + timedelta(hours=float(rng.uniform(0.2, 5)))
            if t1 < row.dischtime:
                report_lag_h = float(rng.lognormal(mean=np.log(0.5), sigma=0.5))
                meas = _ecg_measurements(abnormal=True)
                rows.append(dict(
                    ecg_id=f"{row.subject_id}-ECG{ecg_counter}", subject_id=row.subject_id, hadm_id=row.hadm_id,
                    ecg_time=t1, report_time=t1 + timedelta(hours=report_lag_h),
                    finding=rng.choice(ECG_FINDINGS_ABNORMAL),
                    **meas,
                ))
                ecg_counter += 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5b. Echo events: structured findings, not raw video. Rarer than ECG.
# ---------------------------------------------------------------------------

VALVE_GRADES = ["none", "trace", "mild", "moderate", "severe"]
ECHO_FINDINGS_NORMAL = [
    "Normal biventricular size and systolic function.",
    "Preserved LVEF, no significant valvular disease.",
]
ECHO_FINDINGS_ABNORMAL = [
    "Mild-moderate regional left ventricular systolic dysfunction.",
    "Reduced LVEF with global hypokinesis.",
    "Moderate mitral regurgitation, preserved LVEF.",
]


def generate_echo_events(patients):
    rows = []
    echo_counter = 1
    for row in patients.itertuples():
        p_echo = (0.06 + (0.15 if row.admit_reason in CARDIAC_REASONS else 0)
                  + (0.25 if row.has_cardiac_history else 0)
                  + (0.10 if (row.outcome == 1 and row.hospital_complexity_tier == "C_complex") else 0))
        if rng.random() >= min(p_echo, 0.7):
            continue

        anchor = row.true_onset_time if (row.outcome == 1 and pd.notna(row.true_onset_time) and rng.random() < 0.6) else row.admittime
        t0 = anchor + timedelta(hours=float(rng.uniform(1, 20)))
        if t0 >= row.dischtime:
            t0 = row.admittime + timedelta(hours=float(rng.uniform(1, 10)))
        report_lag_h = float(rng.lognormal(mean=np.log(3.0), sigma=0.5))

        abnormal = rng.random() < (0.20 if row.outcome == 0 else 0.5)
        lvef = round(float(rng.uniform(55, 68) if not abnormal else rng.uniform(25, 45)), 0)
        rows.append(dict(
            echo_id=f"{row.subject_id}-ECHO{echo_counter}", subject_id=row.subject_id, hadm_id=row.hadm_id,
            echo_time=t0, report_time=t0 + timedelta(hours=report_lag_h),
            lvef_percent=lvef,
            wall_motion_abnormality="Y" if abnormal else "N",
            valve_regurgitation=rng.choice(VALVE_GRADES, p=[0.35, 0.25, 0.2, 0.15, 0.05]),
            pericardial_effusion="Y" if rng.random() < (0.08 if not abnormal else 0.2) else "N",
            finding=rng.choice(ECHO_FINDINGS_ABNORMAL) if abnormal else rng.choice(ECHO_FINDINGS_NORMAL),
        ))
        echo_counter += 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5c. Physician interpretation notes for ECG/echo events.
#     A machine/tech report existing is not the same as a clinician having
#     read and charted an interpretation of it. This adds a third timing
#     layer on top of ecg_time -> report_time: report_time -> the moment a
#     clinician actually documents their read, which is the layer that
#     matters clinically and is easy to overlook.
# ---------------------------------------------------------------------------

def generate_diagnostic_interpretation_notes(patients, ecg, echo):
    rows = []
    patients_by_id = patients.set_index("subject_id")
    note_counter = 90000  # separate counter range, avoids colliding with generate_notes' ids

    for study_df, kind, id_col, time_col in [
        (ecg, "ecg_interpretation", "ecg_id", "ecg_time"),
        (echo, "echo_interpretation", "echo_id", "echo_time"),
    ]:
        for e in study_df.itertuples():
            # not every study gets an explicit charted interpretation promptly;
            # some are reviewed later as part of routine rounding, a few are
            # effectively folded into a later note instead
            if rng.random() >= 0.80:
                continue

            prow = patients_by_id.loc[e.subject_id]
            lag_mult = note_lag_multiplier_row(prow, e.subject_id)
            interp_lag_h = float(rng.lognormal(mean=np.log(4.0), sigma=0.6)) * lag_mult
            interp_time = e.report_time + timedelta(hours=interp_lag_h)

            if kind == "ecg_interpretation":
                text = rng.choice(INTERPRETATION_TEMPLATES[kind]).format(finding=e.finding.lower())
            else:
                text = rng.choice(INTERPRETATION_TEMPLATES[kind]).format(finding=e.finding.lower(), lvef=int(e.lvef_percent))

            sev = dementia_severity(prow.population_context)
            rows.append(dict(
                note_id=f"{e.subject_id}-N{note_counter}", subject_id=e.subject_id, hadm_id=e.hadm_id,
                note_type=kind, author_type="clinician",
                event_charttime=getattr(e, time_col), storetime=interp_time,
                text=text, history_reliability="reliable" if sev == 0 else ("limited" if sev >= 2 else "mostly reliable"),
                related_event_id=getattr(e, id_col),
            ))
            note_counter += 1

    return pd.DataFrame(rows)


def note_lag_multiplier_row(prow, subject_id):
    # thin wrapper so this function can reuse note_lag_multiplier on a Series
    # (patients.itertuples() rows and a .loc[] Series both support attribute access)
    return note_lag_multiplier(prow)

def generate_event_log(patients, vitals, labs, notes, ecg, echo):
    rows = []
    for row in patients.itertuples():
        case_id = row.hadm_id
        rows.append(dict(subject_id=row.subject_id, hadm_id=case_id, timestamp=row.admittime, activity="Admission"))
        rows.append(dict(subject_id=row.subject_id, hadm_id=case_id, timestamp=row.admittime, activity="ICU Transfer In"))

        # sample a coarser subset of vitals checks so the log stays log-like, not row-for-row
        pv = vitals[vitals.subject_id == row.subject_id]
        if len(pv):
            step = max(1, len(pv) // 12)
            for t in pv.charttime.iloc[::step]:
                rows.append(dict(subject_id=row.subject_id, hadm_id=case_id, timestamp=t, activity="Vital sign check"))

        pl = labs[labs.subject_id == row.subject_id]
        for specimen_id, grp in pl.groupby("specimen_id"):
            ct = grp.charttime.iloc[0]
            st = grp.storetime.iloc[0]
            rows.append(dict(subject_id=row.subject_id, hadm_id=case_id, timestamp=ct, activity="Lab drawn"))
            rows.append(dict(subject_id=row.subject_id, hadm_id=case_id, timestamp=st, activity="Lab resulted"))

        pn = notes[notes.subject_id == row.subject_id]
        for n in pn.itertuples():
            if n.note_type == "ecg_interpretation":
                act = "ECG interpreted"
            elif n.note_type == "echo_interpretation":
                act = "Echo interpreted"
            elif n.author_type == "caregiver":
                act = "Note authored - caregiver"
            else:
                act = "Note authored - clinician"
            rows.append(dict(subject_id=row.subject_id, hadm_id=case_id, timestamp=n.storetime, activity=act))

        pe = ecg[ecg.subject_id == row.subject_id]
        for e in pe.itertuples():
            rows.append(dict(subject_id=row.subject_id, hadm_id=case_id, timestamp=e.ecg_time, activity="ECG performed"))
            rows.append(dict(subject_id=row.subject_id, hadm_id=case_id, timestamp=e.report_time, activity="ECG reported"))

        pec = echo[echo.subject_id == row.subject_id]
        for e in pec.itertuples():
            rows.append(dict(subject_id=row.subject_id, hadm_id=case_id, timestamp=e.echo_time, activity="Echo performed"))
            rows.append(dict(subject_id=row.subject_id, hadm_id=case_id, timestamp=e.report_time, activity="Echo reported"))

        if row.hospital_complexity_tier == "C_complex":
            mid_t = row.admittime + timedelta(days=row.los_days * float(rng.uniform(0.3, 0.6)))
            rows.append(dict(subject_id=row.subject_id, hadm_id=case_id, timestamp=mid_t, activity="Care unit transfer"))

        rows.append(dict(subject_id=row.subject_id, hadm_id=case_id, timestamp=row.dischtime, activity="Discharge"))

    df = pd.DataFrame(rows)
    return df.sort_values(["hadm_id", "timestamp"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 6. Labels (ground truth, derived from the generated notes for consistency)
# ---------------------------------------------------------------------------

def generate_labels(patients):
    """The instructor's answer key. true_onset_time and the possible_window
    bounds are the generator's internal ground truth, not derivable from any
    raw table (that's the whole point of 'uncertain onset'), so they belong
    here. Anything derivable from raw timestamps (documentation lag, which
    note first captured the event) deliberately does NOT appear here, that's
    Block 1/2 exercise material using synthetic_notes.csv directly."""
    rows = []
    for row in patients.itertuples():
        uncertainty_h = onset_uncertainty_hours(row)
        window_start = window_end = None
        if row.outcome == 1 and pd.notna(row.true_onset_time):
            window_start = row.true_onset_time - timedelta(hours=uncertainty_h / 2)
            window_end = row.true_onset_time + timedelta(hours=uncertainty_h / 2)

        rows.append(dict(
            subject_id=row.subject_id, hadm_id=row.hadm_id,
            outcome=row.outcome, true_onset_time=row.true_onset_time,
            possible_window_start=window_start, possible_window_end=window_end,
            hospital_complexity_tier=row.hospital_complexity_tier,
            entanglement_tier=row.entanglement_tier,
            population_context=row.population_context,
            socioeconomic_context=row.socioeconomic_context,
        ))
    return pd.DataFrame(rows)


def write_medication_formulary(out_path):
    """A simplified, FHIR-inspired medication reference: real EHR systems
    represent this kind of terminology using HL7 FHIR Medication resources
    coded against RxNorm, where one canonical drug concept links together
    every name it might be written under. This mirrors that structure
    without the full resource complexity, deliberately shipped with
    known_synonyms empty, an incompletely-curated dictionary is exactly
    what a real one looks like before anyone has gone looking for the gaps."""
    import json
    formulary = {
        "resourceType": "MedicationFormulary",
        "description": (
            "Simplified, FHIR-inspired medication reference for this tutorial's synthetic corpus. "
            "Real EHR systems solve the brand/generic naming problem with RxNorm-coded FHIR Medication "
            "resources; this file mirrors that structure at a scale appropriate for a three-hour tutorial."
        ),
        "medications": [
            {
                "formulary_id": f"MED-{i+1:03d}",
                "generic_name": med["name"],
                "known_synonyms": [],
                "typical_dose": med["dose"],
                "typical_mode": med["mode"],
                "typical_frequency": med["freq"],
                "common_indications": med["reasons"],
            }
            for i, med in enumerate(MEDICATIONS)
        ],
    }
    with open(out_path, "w") as f:
        json.dump(formulary, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Regenerate the full synthetic corpus from scratch.")
    parser.add_argument("--data-dir", default=".",
                         help="Folder to write the 8 output CSVs and medication_formulary.json into. "
                              "Run with --data-dir ./data if you're running this from the Tutorial/ root. "
                              "WARNING: this overwrites the corpus every notebook reads from, back up "
                              "first if you've since added amendments or anything else built on top of it.")
    args = parser.parse_args()
    OUT_DIR = args.data_dir

    patients = generate_patients()          # full internal frame: has outcome, true_onset_time, los_days
    vitals = generate_vitals(patients)
    labs = generate_labs(patients)
    notes_core = generate_notes(patients)
    notes_core, med_labels = add_medication_mentions(notes_core, patients)
    ecg = generate_ecg_events(patients)
    echo = generate_echo_events(patients)
    interp_notes = generate_diagnostic_interpretation_notes(patients, ecg, echo)
    notes = pd.concat([notes_core, interp_notes], ignore_index=True).sort_values(["subject_id", "storetime"]).reset_index(drop=True)
    event_log = generate_event_log(patients, vitals, labs, notes, ecg, echo)
    labels = generate_labels(patients)      # the answer key: outcome + true_onset_time live ONLY here going forward

    # Public patients.csv drops los_days (computable from admit/disch), outcome and
    # true_onset_time (the ground truth belongs in labels.csv only, not leaked
    # into the file participants build the Block 1 timeline exercise from).
    # comorbidities / has_cardiac_history stay: that's real observable patient
    # history, not the hidden ground truth, and it's what actually drives
    # whether a given patient has ECG/echo events at all.
    patients_public = patients.drop(columns=["los_days", "outcome", "true_onset_time"])

    print("patients:", patients_public.shape)
    print("vitals:", vitals.shape)
    print("labs:", labs.shape)
    print("notes:", notes.shape, " (of which interpretation notes:", len(interp_notes), ")")
    print("medication_labels:", med_labels.shape)
    print("ecg:", ecg.shape)
    print("echo:", echo.shape)
    print("event_log:", event_log.shape)
    print("labels:", labels.shape)

    round_datetimes(patients_public).to_csv(f"{OUT_DIR}/synthetic_patients.csv", index=False)
    round_datetimes(vitals).to_csv(f"{OUT_DIR}/synthetic_vitals_continuous.csv", index=False)
    round_datetimes(labs).to_csv(f"{OUT_DIR}/synthetic_labs_intermittent.csv", index=False)
    round_datetimes(notes).to_csv(f"{OUT_DIR}/synthetic_notes.csv", index=False)
    med_labels.to_csv(f"{OUT_DIR}/synthetic_medication_labels.csv", index=False)
    round_datetimes(ecg).to_csv(f"{OUT_DIR}/synthetic_ecg_events.csv", index=False)
    round_datetimes(echo).to_csv(f"{OUT_DIR}/synthetic_echo_events.csv", index=False)
    round_datetimes(event_log).to_csv(f"{OUT_DIR}/synthetic_event_log.csv", index=False)
    round_datetimes(labels).to_csv(f"{OUT_DIR}/synthetic_labels.csv", index=False)
    write_medication_formulary(f"{OUT_DIR}/medication_formulary.json")
    print("done")
