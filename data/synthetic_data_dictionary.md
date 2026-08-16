# Synthetic Data Dictionary

75 synthetic patients, no real patient data anywhere. Built from `synthetic_generator.py`, calibrated against timing statistics pulled from the uploaded MIMIC-derived samples (not against any individual real record). Two layers: a **relational layer** (nine CSVs) and a **document layer** (one schema-validated JSON per admission), plus `viz.py` for Block 1.

All CSVs key on `subject_id` (patient) and `hadm_id` (admission/case). Join freely.

## Design principle: raw files stay raw

Nothing that requires comparing two timestamps, documentation lag, time-to-result, time-to-interpretation, is precomputed anywhere in the output tables. That comparison is the Block 1/2 exercise. The one exception is `synthetic_labels.csv`, the instructor's answer key: it carries `true_onset_time` and the uncertainty band around it, values no real hospital system records directly and that participants are meant to be inferring, not reading off a column.

## Design axes

| Axis | Values | Controls |
|---|---|---|
| `hospital_complexity_tier` | A_minimal / B_moderate / C_complex | length of stay, vitals sampling rate, labs per day, note count |
| `entanglement_tier` | low / medium / high | note lag, lab lag, onset uncertainty, missingness |
| `population_context` | none / neurodivergent / disability / dementia_mild / dementia_moderate / dementia_severe | see mechanism table below |
| `socioeconomic_context` | standard_resourced / under_resourced | missingness +0.10, note lag ×1.3, independent of the other three |
| `comorbidities` / `has_cardiac_history` | free-text list / boolean | drives ECG and echo ordering probability directly |

### Population context mechanisms (each modifies a *different* failure mode)

| Context | Mechanism | What it touches |
|---|---|---|
| neurodivergent | wider baseline physiological variance (×1.6 noise) | `synthetic_vitals_continuous.csv` |
| disability | degraded signal / equipment fit (+0.15 missingness) + communication lag (×1.3) | vitals + notes |
| dementia (mild/moderate/severe) | widening onset uncertainty (×1.3/1.8/2.5), self-report reliability downgrade, caregiver-note probability rising to guaranteed | labels + notes |
| any disability/dementia | ~55% chance an event note is "overshadowed" (attributed to baseline condition) with an extra 1.4× lag penalty | notes + labels |

These stack. A dementia_severe + under_resourced patient compounds three mechanisms at once, on purpose.

## Relational layer (CSVs)

### `synthetic_patients.csv` (75 rows)
`subject_id, hadm_id, stay_id, age, gender, careunit, admittime, dischtime, hospital_complexity_tier, entanglement_tier, population_context, socioeconomic_context, admit_reason, comorbidities, has_cardiac_history`.
No `outcome` or `true_onset_time` here on purpose, that's the answer key, it lives only in `synthetic_labels.csv`.

### `synthetic_vitals_continuous.csv` (~12,200 rows)
`heartrate, resprate, o2sat, sbp, dbp` at a tier-dependent interval (15/30/60 min). `signal_quality` flags degraded/missing readings.

### `synthetic_labs_intermittent.csv` (~2,100 rows)
Five standard labs (Lactate, Potassium, Creatinine, WBC, Troponin). `charttime` (drawn) vs `storetime` (resulted) is the lab-lag pair, left uncomputed.

### `synthetic_notes.csv` (~360 rows)
The delayed-narrative stream: clinician, caregiver, *and* diagnostic-interpretation notes, unified with `note_type`/`author_type`. `event_charttime` vs `storetime` is the documentation-lag pair. `related_event_id` links `ecg_interpretation`/`echo_interpretation` rows back to the ECG/echo files. `history_reliability` flags self-report confidence for dementia contexts. A subset of admission notes (weighted toward cardiac-history patients) end with a generated medication sentence, see below.

### `synthetic_medication_labels.csv` (~205 rows) — **new this round**
Character-span annotations mirroring the real `medication-labels-mimic-note` dataset exactly: `note_id, Start Position, End Position, Annotation, Group`. `Annotation` is one of MEDICATION / DOSAGE / MODE / FREQUENCY / REASON. Spans are exact by construction (we wrote the text ourselves), so `note.text[start:end]` always recovers the labeled substring, participants get the same span-to-text join workflow described in the real dataset's README, just on safe synthetic text. Validated: 205/205 spans extract exactly; mention rate 67.6% for cardiac-history patients vs 42.1% for everyone else.

### `synthetic_ecg_events.csv` (~55 rows) / `synthetic_echo_events.csv` (~22 rows)
Structured interval/axis measurements (ECG, mirroring `Master_Sheet_Sample.csv`'s real schema) and structured findings (echo: LVEF, wall motion, valve grade), not raw waveform or video. Three-stage timing: performed → technical report → a separate clinician interpretation note in `synthetic_notes.csv` (linked via `related_event_id`), each stage with its own lag. Ordering probability driven by `has_cardiac_history` and `admit_reason`. Echo is not calibrated against any real sample, we never located one; ECG is.

### `synthetic_event_log.csv` (~2,900 rows)
mimicel-style: `hadm_id` as case ID, `activity`, `timestamp`. Distinguishes "ECG interpreted" / "Echo interpreted" from generic "Note authored - clinician" so the three-stage diagnostic chain is visible directly in the log.

### `synthetic_labels.csv` (75 rows)
`outcome, true_onset_time, possible_window_start/end`, plus all tier/context columns for subgroup evaluation. The only file with the ground truth in it.

## Document layer — **new this round**

### `document_schema.json`
A formal JSON Schema (2020-12) defining `TemporallyEntangledDocument`: one admission as a single self-contained object rather than relational rows, with embedded provenance on every evidence item. This is the piece that makes the tutorial actually about document representation, not just tabular data.

### `synthetic_documents.json`
All 75 admissions assembled per the schema. **Validated: 75/75 pass** `jsonschema.validate()`. Each document embeds its full vitals series, all lab draws, all notes (with medication spans inlined), and any ECG/echo studies. Self-contained by design, one file is one complete patient story; the tradeoff is size (~4MB total for 75 patients, since vitals are embedded rather than referenced). Happy to switch to a summary-plus-pointer pattern for the continuous stream if you'd rather keep documents smaller, just say so.

### `document_representation.py`
Builds the above, and demonstrates the pipeline step that actually matters for document engineering: structured sources → unified document model → generated human-readable output. Run directly, it also renders one example admission to `example_chart_<hadm_id>.md`, a chart summary generated *from* the structured document, not hand-written.

## Visualization

### `viz.py`
Block 1's module: `plot_patient_timeline` (Figure-1-style multi-track view: continuous signal / intermittent labs / delayed notes, now a fourth row when ECG/echo are present, each with connector lines showing event-time → charted-time gaps), `plot_event_window` (the uncertainty-band diagram), `plot_lag_by_tier` (aggregate lag by entanglement tier, computed live). Nothing here trains a model, that's Block 2.

## Validated on this generation

Referential integrity (0 orphaned IDs across every file, including `related_event_id` and medication-label `note_id` links), timestamp ordering (0 violations, including the full three-stage ECG chain and all 205 medication spans), documentation lag scaling monotonically with entanglement tier, onset uncertainty widening with dementia severity, neurodivergent vitals noise elevated as designed, ECG ordering rate higher for cardiac-history patients (78% vs 50%), and all 75 documents passing schema validation.

## Still to come

Medication is back in (this round). Not yet built: **Block 3's temporal-inconsistency model audit tool** (using a mismatch between when evidence became available and when a model used it as an explicit audit signal, distinct from ordinary attribution), `alignment.py`, `models.py`, and the domain-swap configs (legal/education/audit). Queued, in that rough order.
