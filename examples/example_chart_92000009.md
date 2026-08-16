# Admission 92000009  (patient 90000009)

**84yo F**, CVICU, admitted for *altered mental status*
2025-09-24T06:12:00 → 2025-09-25T16:38:00

**Comorbidities:** atrial fibrillation

**Design context:** B_moderate · high entanglement · dementia_moderate · standard_resourced

## Narrative timeline

**[admission]** (clinician) — event: `2025-09-24T06:12:00`, charted: `2025-09-24T12:27:00`
> Patient admitted for altered mental status. Baseline vitals reviewed. Plan: monitor and treat per protocol. Case discussed with the primary team. Patient and family updated on initial plan. Plan: start albuterol 2.5mg, nebulized, every 4 hours as needed for dyspnea. Baseline labs sent, results pending. No acute distress noted at this time.
medication spans: MEDICATION=“albuterol”, DOSAGE=“2.5mg”, MODE=“nebulized”, FREQUENCY=“every 4 hours as needed”, REASON=“dyspnea”

**[event_clinician]** (clinician) — event: `2025-09-24T18:26:00`, charted: `2025-09-24T23:18:00`
> Nursing reports patient appears more unwell since last check. Reassessed, orders updated accordingly.

**[ecg_interpretation]** (clinician) — event: `2025-09-24T08:25:00`, charted: `2025-09-25T08:36:00`
> ECG reviewed: sinus rhythm, otherwise normal ecg. Clinically correlated with current presentation, plan unchanged.

**[ecg_interpretation]** (clinician) — event: `2025-09-24T20:12:00`, charted: `2025-09-25T14:41:00`
> Reviewed today's ECG, left ventricular hypertrophy noted. Will trend with serial tracings.

**[routine_stable]** (clinician) — event: `2025-09-24T18:27:00`, charted: `2025-09-25T21:23:00`
> Overnight course unremarkable. Vitals within patient's usual range. Continue monitoring. Physical therapy assessment completed today. Home medication metoprolol (25mg, PO, BID) continued for rate control. Sleeping well overnight per patient report.
medication spans: MEDICATION=“metoprolol”, DOSAGE=“25mg”, MODE=“PO”, FREQUENCY=“BID”, REASON=“rate control”

**[discharge]** (clinician) — event: `2025-09-25T16:38:00`, charted: `2025-09-26T10:38:00`
> Patient's condition improved / stabilized over the admission. Discharged with follow-up plan in place. Transportation home arranged with family. Follow-up appointment scheduled with primary care. Discharge medications include heparin 5000 units subcutaneous every 8 hours for DVT prophylaxis. Wound care instructions provided, if applicable. Discharge instructions reviewed with patient and family.
medication spans: MEDICATION=“heparin”, DOSAGE=“5000 units”, MODE=“subcutaneous”, FREQUENCY=“every 8 hours”, REASON=“DVT prophylaxis”

**[AMENDMENT → corrects 90000009-N33]** (clinician) — event: `2025-09-24T18:27:00`, charted: `2025-09-26T13:48:00` — *dosage transcription error*
> ADDENDUM to note 90000009-N33: dosage was charted incorrectly. Corrected dosage: 12.5mg (previously 25mg).
medication spans: DOSAGE=“12.5mg”

## ECG studies

- 2025-09-24T08:25:00: Normal sinus rhythm (reported 2025-09-24T08:57:00)
- 2025-09-24T20:12:00: Left ventricular hypertrophy (reported 2025-09-24T20:36:00)

## Summary counts

- 68 continuous vitals readings
- 3 lab draws
- 7 notes