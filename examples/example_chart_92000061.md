# Admission 92000061  (patient 90000061)

**71yo F**, MICU, admitted for *chest pain*
2024-01-07T15:54:00 → 2024-01-12T18:18:00

**Design context:** C_complex · low entanglement · dementia_severe · under_resourced

## Narrative timeline

**[admission]** (clinician) — event: `2024-01-07T15:54:00`, charted: `2024-01-07T21:19:00`
> Patient admitted for chest pain. Baseline vitals reviewed. Plan: monitor and treat per protocol.

**[ecg_interpretation]** (clinician) — event: `2024-01-07T17:51:00`, charted: `2024-01-08T04:19:00`
> Reviewed today's ECG, atrial fibrillation noted. Will trend with serial tracings.

**[caregiver]** (caregiver) — event: `2024-01-08T23:03:00`, charted: `2024-01-09T00:37:00`
> Caregiver present at bedside, notes patient is more unsettled than is typical for them.

**[event_overshadowed]** (clinician) — event: `2024-01-08T21:35:00`, charted: `2024-01-09T06:22:00`
> Change in presentation noted, consistent with patient's known baseline condition. No acute intervention indicated at this time; continue current plan.

**[routine_stable]** (clinician) — event: `2024-01-09T10:42:00`, charted: `2024-01-09T18:35:00`
> No acute changes overnight. Patient stable, tolerating current plan of care. Sleeping well overnight per patient report. Pain well controlled on current regimen. Home medication apixaban (5mg, PO, BID) continued for atrial fibrillation. Input and output balanced over the last shift. Overnight vitals within the patient's acceptable range.
medication spans: MEDICATION=“apixaban”, DOSAGE=“5mg”, MODE=“PO”, FREQUENCY=“BID”, REASON=“atrial fibrillation”

**[echo_interpretation]** (clinician) — event: `2024-01-09T14:19:00`, charted: `2024-01-10T03:41:00`
> Echo reviewed: reduced lvef with global hypokinesis. LVEF 37%. Findings incorporated into management plan.

**[routine_stable]** (clinician) — event: `2024-01-09T18:16:00`, charted: `2024-01-10T09:21:00`
> No acute changes overnight. Patient stable, tolerating current plan of care.

**[routine_stable]** (clinician) — event: `2024-01-11T02:11:00`, charted: `2024-01-11T08:12:00`
> No acute changes overnight. Patient stable, tolerating current plan of care.

**[routine_stable]** (clinician) — event: `2024-01-12T06:41:00`, charted: `2024-01-12T11:00:00`
> No acute changes overnight. Patient stable, tolerating current plan of care. No new complaints reported by patient this shift. Mobility and appetite both improving steadily. Continues on levofloxacin 500mg IV daily for presumed infection, tolerating well. Pain well controlled on current regimen. Overnight vitals within the patient's acceptable range.
medication spans: MEDICATION=“levofloxacin”, DOSAGE=“500mg”, MODE=“IV”, FREQUENCY=“daily”, REASON=“presumed infection”

**[AMENDMENT → corrects 90000061-N246]** (clinician) — event: `2024-01-12T06:41:00`, charted: `2024-01-12T14:14:00` — *wrong drug charted, corrected to actual order*
> ADDENDUM to note 90000061-N246: medication was charted incorrectly. Corrected medication: albuterol (previously levofloxacin).
medication spans: MEDICATION=“albuterol”

**[discharge]** (clinician) — event: `2024-01-12T18:18:00`, charted: `2024-01-13T15:26:00`
> Patient's condition improved / stabilized over the admission. Discharged with follow-up plan in place.

## ECG studies

- 2024-01-07T17:51:00: Sinus bradycardia (reported 2024-01-07T19:08:00)

## Echo studies

- 2024-01-09T14:19:00: Reduced LVEF with global hypokinesis., LVEF 37.0% (reported 2024-01-09T17:04:00)

## Summary counts

- 489 continuous vitals readings
- 20 lab draws
- 11 notes