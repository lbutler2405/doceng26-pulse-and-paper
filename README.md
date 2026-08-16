# Temporally Entangled Documents

### Multimodal AI for ICU Records Under Label Ambiguity
**A DocEng '26 Tutorial · Fribourg, Switzerland**

Liam Butler · Department of Systems and Control Engineering, University of Malta · [liam.butler@um.edu.mt](mailto:liam.butler@um.edu.mt)

---

## Start here

Everything runs in Google Colab. Nothing to install, nothing to configure. You need a Google account and a browser tab.

| Block | What it covers | Open |
|---|---|---|
| **0** | Understanding the Data | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lbutler2405/doceng26-pulse-and-paper/blob/main/Notebooks/block0_understanding_the_data.ipynb) |
| **1** | Meet Your Patient | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lbutler2405/doceng26-pulse-and-paper/blob/main/Notebooks/block1_meet_your_patient.ipynb) |
| **2** | Catch the Model Cheating | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lbutler2405/doceng26-pulse-and-paper/blob/main/Notebooks/block2_catch_the_model_cheating.ipynb) |
| **3** | Explainability and Generalisation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lbutler2405/doceng26-pulse-and-paper/blob/main/Notebooks/block3_explainability_and_generalization.ipynb) |

Three things to know before you run anything:

1. **Colab will warn you that the notebook was not authored by Google.** That appears for any notebook opened from GitHub. Click **Run anyway**.
2. **Run the first cell before anything else.** It starts with `!git clone` and pulls this repository into your Colab session. Everything after it depends on that.
3. **Work top to bottom.** Shift+Enter runs a cell and moves to the next. Later cells depend on earlier ones having run.

If you would rather run it locally, clone the repository and open the notebooks in Jupyter. The only change needed is `DATA_DIR` in the first cell, which should point at your local copy instead of the Colab path.

---

## What this is about

Picture a patient's ICU chart. A heart monitor traces a continuous line. A nurse logs a blood pressure reading every hour. A doctor writes a note at 4pm describing a deterioration this morning.

Three sources, three timelines, one patient. Try to line them up and the chart stops looking like a record and starts looking like a puzzle whose pieces disagree about what time it is.

I call these **temporally entangled document systems**: layered, asynchronous evidence streams whose meaning depends as much on when something was recorded as on what it says. Four failure modes keep surfacing.

- **Asynchronous evidence.** A biosignal streams continuously, labs arrive in bursts, a note is written once about a stretch of time already past. None of these clocks agree.
- **Documentation lag.** A note written at 4pm may describe an event from 10am. Treat it as though it were written at 10am and your model learns the shape of hospital paperwork rather than the shape of the disease.
- **Uncertain onset.** Deterioration and most diagnoses worth catching early have no single ground-truth timestamp. They emerge gradually and get named in hindsight.
- **A record that changes after the fact.** A charted note gets corrected a day later. The chart said one thing on Tuesday and something else on Wednesday, and both were true when you asked.

None of this is unique to hospitals. Legal proceedings, student records, and financial audit trails all entangle what was recorded with when it was recorded. The ICU is where the stakes and the mess are both easiest to see, which is why I use it as the test bed.

---

## What the tutorial covers

Three hours, four blocks, hands-on throughout. No clinical background needed and no confident coding needed. Every cell is written and tested. You run it, look at what comes out, and change things to see what moves.

### Block 0 · Understanding the Data
A corpus-wide tour before anything gets modelled. Where each evidence stream comes from in a real hospital, and what real export formats actually look like. You write a naive medication dictionary, watch it catch about 83% of mentions, find that every miss is a brand name nobody taught it, extend it, and rerun for full recall. You open a raw MUSE ECG XML and a raw HL7 FHIR bundle, parse both, and decode ten seconds of base64-encoded waveform into an actual plotted heartbeat.

### Block 1 · Meet Your Patient
The corpus narrows to one chart. You build the multi-track timeline, with every note drawn twice, once for when it happened and once for when it was charted. You compute documentation lag by hand, which is the part I will not let the software do for you. You watch CSV rows, a parsed ECG, and parsed FHIR data unify into one schema-validated document. Then amendments: a note corrected after the fact, and a schema that rejects a correction which fails to say what it corrects. The block ends with two PDFs of the same patient that disagree with each other on purpose.

### Block 2 · Catch the Model Cheating
Three alignment techniques used by hand first: event-centred windowing, soft labelling under uncertain onset, and structured missingness. Then the obvious model, everything pooled into one fixed window. It scores excellently and nothing looks wrong. Read its coefficients and a documentation-timing feature carries real weight. Isolate that signal with all physiology removed and it still clears a coin flip, which is exactly why a model reaches for it. Split by patient group and its recall swings from roughly a quarter to nearly ninety percent under identical rules. The fix is architectural, same pipeline with different features, and the block closes with an evaluation report you write part of yourself.

### Block 3 · Explainability and Generalisation
Exact per-prediction attribution, checked against the model's own output rather than approximated. Then all four evidence streams, biosignal, tabular, diagnostic studies, and the content of the notes, with contributions grouped by modality and set beside how old each stream's evidence was at prediction time. A reusable audit tool that flags predictions resting on stale evidence. An honest account of why every AUROC in this tutorial is 1.000, with the generator parameter responsible. A mapping from EU AI Act requirements to artefacts already built. And then the same alignment functions, imported unchanged, running on a corpus of students in an online course.

---

## What a record actually looks like

Rather than describe the problem, here is one. Patient 90000009: 84 years old, admitted with altered mental status, high entanglement tier, moderate dementia. This is a real rendered chart from the corpus, produced by `document_representation.py` from the assembled document.

Read the two timestamps on each note before anything else.

<details>
<summary><b>Click to expand the full chart</b></summary>

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


</details>

**Four things to notice, all of them in that one chart.**

**The lag is not constant.** The admission note was charted 6.2 hours after the event it describes. The routine note took 26.9 hours. The ECG interpretation took 24.2 hours, despite the ECG itself having been reported within about thirty minutes of being taken. A model that treats a note's charted time as the time something happened is wrong by a different amount every single time.

**The charting order is not the event order.** Sorted by when things happened, the sequence runs admission, ECG interpretation, clinician event, routine note. Sorted by when they were typed, the ECG interpretation lands after the clinician event instead. Read the chart top to bottom in charted order and you get a different story about this stay than the one that actually occurred.

**The record changed after discharge.** The last entry is an amendment, charted on the 26th, correcting a metoprolol dose from 25mg to 12.5mg in a note written on the 24th. Its event time is copied from the note it corrects, which is why it carries a 43.4 hour lag, the largest in the chart. Ask what dose this patient was on and the honest answer depends on when you asked.

**Structure was recovered from prose.** The medication spans under each note are character offsets into the free text, not a separate structured field. Drug, dose, route, frequency, and reason, pulled out of a sentence a clinician typed.

Every one of those is a document engineering problem before it is a machine learning problem, and that is the argument the tutorial spends three hours making.

### More examples

Five rendered charts are in [`examples/`](examples/) if you would like to compare. They are deliberately different from each other.

| Chart | Why it is interesting |
|---|---|
| [`example_chart_92000009.md`](examples/example_chart_92000009.md) | The one above. Highest complexity in the corpus, seven notes, an amendment, dementia context. |
| [`example_chart_92000061.md`](examples/example_chart_92000061.md) | The longest chart, eleven notes. Severe dementia and an under-resourced setting, with a caregiver voice in the record. |
| [`example_chart_92000004.md`](examples/example_chart_92000004.md) | Assembled from a real FHIR bundle rather than CSV, and carries an amendment that swaps one drug for another rather than correcting a dose. |
| [`example_chart_92000006.md`](examples/example_chart_92000006.md) | Real FHIR and real MUSE behind it, with both an ECG and an echo. |
| [`example_chart_92000002.md`](examples/example_chart_92000002.md) | A deliberately plain one. Three notes, no amendment, medium entanglement. Useful as a baseline for what the others are departing from. |


---

## The data

Everything is synthetic. Seventy-five patients, generated to reflect the timing patterns of real clinical records without containing any real patient's information. Timing parameters are calibrated against public MIMIC-derived statistics for lab delays, documentation lag, and sampling intervals.

Three complexity axes shape each admission: how much data the stay generates, how entangled its timing is, and a population context that changes a different failure mode for each group. A socioeconomic flag compounds on top.

Underneath a subset of patients sit **real raw export files**: 10 MUSE-format ECG XMLs and 6 HL7 FHIR R4 bundles. These are parsed by code that genuinely reads them, and the provenance in the assembled documents reflects what actually happened at parse time rather than a label applied afterwards.

A second, separate corpus of 20 students in an online course exists for one purpose: to let Block 3 demonstrate generalisation rather than assert it.

---

## Repository layout

```
├── Notebooks/                  the four tutorial blocks
├── data/
│   ├── synthetic_*.csv         the nine-CSV relational layer
│   ├── synthetic_documents.json   all 75 admissions, assembled and schema-validated
│   ├── document_schema.json    the formal document definition
│   ├── education_*.csv         the second-domain corpus for Block 3
│   └── XML_JSON/
│       ├── muse_xml/           real MUSE-format ECG exports
│       └── fhir_export/        real HL7 FHIR R4 bundles
└── *.py                        the modules the notebooks import
```

**The modules**

| File | What it does |
|---|---|
| `synthetic_generator.py` | Generates the whole corpus. Every timing parameter lives here. |
| `generate_amendments.py` | Adds the amendment and versioning layer to the notes. |
| `document_representation.py` | Assembles the CSVs, parsed ECGs, and parsed FHIR into schema-validated documents. |
| `parse_muse_xml.py` / `parse_fhir_bundle.py` | Read the real export formats, including waveform decoding. |
| `alignment.py` | Event-centred windowing, soft labelling, structured missingness. Domain-agnostic. |
| `models.py` | The naive and time-aware feature builders and the shared pipeline. |
| `viz.py` | Multi-track timelines, event windows, lag-by-tier plots. |
| `pdf_report.py` / `model_report.py` | The participant-generated PDF artefacts. |
| `generate_education_domain.py` | Builds the second-domain corpus. |

To regenerate the corpus from scratch, run these in order from the repository root:

```bash
python synthetic_generator.py --data-dir ./data
python generate_amendments.py --data-dir ./data
python document_representation.py --data-dir ./data
```

Every number in the notebooks changes if you do. That is the point of the exercise rather than a problem with it.

---

## A note on the AUROC

Every model in this tutorial scores an AUROC of 1.000, and Block 3 explains why rather than leaving it to be discovered. `DETERIORATION_SHIFT` in the generator applies a heart rate change of +38 bpm against a baseline standard deviation of 6. That is more than six standard deviations, so the two outcome groups cannot overlap and a single threshold classifies every patient correctly.

No AUROC here should be read as evidence that a model is good. None of the tutorial's actual findings rest on it either. The shortcut model's 0.562, the subgroup recall swing, the flagged predictions in the audit, and the Brier score gap all move. Lowering the shift or raising the baseline variance produces realistic overlap, and I have left that as an exercise.

---

## Requirements

The notebooks install what they need. If you are running locally: Python 3.9 or later, plus `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `reportlab`, and `jsonschema`.

---

## Ethics

No real patient data is loaded, processed, or displayed at any point. The corpus is synthetic throughout, and the real export files are synthetic records written in genuine export formats rather than real records in any form.

---

## Acknowledgments

This work is part of CAIRED: Cardiovascular Artificial Intelligence, e-Health for Diabetes, funded by Beating Hearts Malta through RIDT.

Thanks to Dr. Kenneth Scerri for his collaboration and unwavering support.

---

## License

The synthetic data and tutorial materials are provided for educational use. Please get in touch if you would like to adapt them for teaching elsewhere, I would genuinely like to hear about it.
