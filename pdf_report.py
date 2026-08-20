"""
pdf_report.py

  render_naive_pdf(patient_id): what most systems would hand you.
      Every fact on one page, one timestamp each, charted order. This is
      the artifact that makes it look like the ICU stay unfolded exactly
      as written down. It's plausible. It's also quietly wrong about time:
      a note charted very late looks identical to one charted on time,
      and a lab's "date" doesn't say whether it's when blood was drawn or
      when the result came back.

  render_engineered_pdf(patient_id): the same evidence, engineered.
      Every fact now carries both timestamps that actually exist in the
      record (when it happened, when it was charted) plus the lag between
      them. The possible deterioration window is shown rather than hidden.
      The narrative is ordered by what happened, not by who got to a
      keyboard first. Medication detail comes from the exact character
      spans extracted earlier in the notebook, not eyeballed. You choose
      which sections go in via the `sections` dict.
"""

import os
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import LETTER
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable
)

from viz import load_patient_bundle, plot_patient_timeline, TEAL, CORAL, AMBER, DARK, MUTED
from generate_amendments import resolve_as_of  # noqa: F401  (re-exported for convenience)

C_TEAL = colors.HexColor(TEAL)
C_CORAL = colors.HexColor(CORAL)
C_AMBER = colors.HexColor(AMBER)
C_DARK = colors.HexColor(DARK)
C_MUTED = colors.HexColor(MUTED)
C_LIGHT_TEAL = colors.HexColor("#EAF5F4")
C_LIGHT_AMBER = colors.HexColor("#FBF3E1")
C_GRID = colors.HexColor("#DDDDDD")
C_ZEBRA = colors.HexColor("#F7F7F7")

DEFAULT_SECTIONS = dict(
    demographics=True, vitals_chart=True, labs=True,
    notes_timeline=True, medications=True, ecg=True, echo=True,
)


def _styles():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle("ReportTitle", parent=ss["Title"], textColor=C_DARK, fontSize=18,
                           spaceAfter=2, alignment=0))
    ss.add(ParagraphStyle("ReportSubtitle", parent=ss["Normal"], textColor=C_MUTED, fontSize=9.5,
                           spaceAfter=12))
    ss.add(ParagraphStyle("Section", parent=ss["Heading2"], textColor=C_TEAL, fontSize=13,
                           spaceBefore=14, spaceAfter=6))
    ss.add(ParagraphStyle("Body", parent=ss["Normal"], fontSize=9.5, leading=13, spaceAfter=4))
    ss.add(ParagraphStyle("NoteBody", parent=ss["Normal"], fontSize=9, leading=12.5, spaceAfter=10))
    ss.add(ParagraphStyle("Callout", parent=ss["Normal"], fontSize=9.5, leading=13,
                           backColor=C_LIGHT_AMBER, borderColor=C_AMBER, borderWidth=0.75,
                           borderPadding=8, spaceBefore=4, spaceAfter=10))
    ss.add(ParagraphStyle("Small", parent=ss["Normal"], fontSize=8, textColor=C_MUTED, leading=11))
    return ss


def _table(rows, col_widths, header_bg=C_MUTED):
    t = Table(rows, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.4, C_GRID),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, C_ZEBRA]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


# Stage A: the naive render

def render_naive_pdf(patient_id, data_dir=".", save_path=None):
    """One timestamp per fact, charting order throughout. This is what a
    quick export looks like when nobody has engineered the document to
    distinguish 'when it happened' from 'when someone wrote it down'."""
    b = load_patient_bundle(patient_id, data_dir)
    patient, vitals, labs, notes = b["patient"], b["vitals"], b["labs"], b["notes"]
    ss = _styles()
    save_path = save_path or f"naive_summary_{patient_id}.pdf"

    doc = SimpleDocTemplate(save_path, pagesize=LETTER, topMargin=0.75 * inch,
                             bottomMargin=0.75 * inch, leftMargin=0.75 * inch, rightMargin=0.75 * inch)
    story = [
        Paragraph("Patient Summary", ss["ReportTitle"]),
        Paragraph(f"MRN {patient_id}  \u00b7  Generated {datetime.now():%d %b %Y}", ss["ReportSubtitle"]),
        HRFlowable(width="100%", color=C_MUTED, thickness=0.5),
        Spacer(1, 10),
    ]

    story.append(Paragraph("Vitals", ss["Section"]))
    if len(vitals):
        story.append(Paragraph(
            f"Heart rate averaged {vitals.heartrate.mean():.0f} bpm over the stay, "
            f"O2 sat averaged {vitals.o2sat.mean():.0f}%.", ss["Body"]))
    else:
        story.append(Paragraph("No continuous signal on file.", ss["Body"]))

    story.append(Paragraph("Labs", ss["Section"]))
    if len(labs):
        rows = [["Date", "Test", "Result"]]
        for _, r in labs.sort_values("charttime").iterrows():
            rows.append([r.charttime.strftime("%d %b"), r.label, r.flag if pd.notna(r.flag) else "normal"])
        story.append(_table(rows, [1.2 * inch, 2.6 * inch, 2.4 * inch]))
    else:
        story.append(Paragraph("No labs on file.", ss["Body"]))

    story.append(Paragraph("Clinical Course", ss["Section"]))
    if len(notes):
        for _, n in notes.sort_values("storetime").iterrows():
            story.append(Paragraph(f"<b>{n.storetime.strftime('%d %b %Y')}</b> \u2014 {n.text}", ss["NoteBody"]))
    else:
        story.append(Paragraph("No notes on file.", ss["Body"]))

    doc.build(story)
    return save_path


# Stage B: the engineered render

def render_engineered_pdf(patient_id, data_dir=".", sections=None, save_path=None, work_dir=".", as_of_time=None):
    """Same evidence as render_naive_pdf, but every fact carries both of its
    real timestamps, the possible event window is shown, and the narrative
    is ordered by when things happened. `sections` lets you switch parts
    on/off, e.g. sections={"ecg": False} to skip a section entirely.

    as_of_time: if given, renders the document exactly as it would have
    read at that moment, amendments charted after as_of_time simply
    haven't happened yet from the document's point of view, and the note
    they'd have corrected still shows its original, uncorrected value.
    Leave as None for the full current state, corrections included."""
    sections = {**DEFAULT_SECTIONS, **(sections or {})}
    b = load_patient_bundle(patient_id, data_dir)
    patient, vitals, labs, notes, ecg, echo, label = (
        b["patient"], b["vitals"], b["labs"], b["notes"], b["ecg"], b["echo"], b["label"]
    )
    has_amendments = "is_amendment" in notes.columns and "amends_note_id" in notes.columns
    if has_amendments and as_of_time is not None:
        notes = notes[notes.storetime <= pd.Timestamp(as_of_time)].copy()

    ss = _styles()
    save_path = save_path or f"engineered_report_{patient_id}.pdf"

    doc = SimpleDocTemplate(save_path, pagesize=LETTER, topMargin=0.75 * inch,
                             bottomMargin=0.75 * inch, leftMargin=0.75 * inch, rightMargin=0.75 * inch)
    story = [
        Paragraph("Clinical Timeline \u2014 Document-Engineered Summary", ss["ReportTitle"]),
        Paragraph(
            f"Patient {patient_id}  \u00b7  {patient.hospital_complexity_tier}, "
            f"{patient.entanglement_tier} entanglement  \u00b7  Generated {datetime.now():%d %b %Y}",
            ss["ReportSubtitle"]),
        HRFlowable(width="100%", color=C_TEAL, thickness=1),
        Spacer(1, 8),
    ]

    if as_of_time is not None:
        story.append(Paragraph(
            f"<b>This document reflects the record AS OF {pd.Timestamp(as_of_time):%d %b %Y %H:%M}</b>, "
            f"not its current state. Any correction charted after this moment is not yet visible below, "
            f"because it wasn't yet visible to anyone reading the chart at this moment either.",
            ss["Callout"]))

    if sections.get("demographics", True):
        story.append(Paragraph("Demographics", ss["Section"]))
        story.append(Paragraph(
            f"Admitted {patient.admittime:%d %b %Y %H:%M}, discharged {patient.dischtime:%d %b %Y %H:%M}. "
            f"Population context: {patient.population_context}. "
            f"Socioeconomic context: {patient.socioeconomic_context}.", ss["Body"]))

    if pd.notna(label.true_onset_time):
        window_h = (label.possible_window_end - label.possible_window_start).total_seconds() / 3600
        story.append(Spacer(1, 6))
        story.append(Paragraph(
            f"<b>Possible deterioration window:</b> {label.possible_window_start:%d %b %H:%M} "
            f"\u2013 {label.possible_window_end:%d %b %H:%M} ({window_h:.1f}h wide). True onset is "
            f"not directly observed in the record \u2014 this window is the evidence-based estimate, "
            f"the same one the model in Block 2 will have to work with.", ss["Callout"]))

    if sections.get("vitals_chart", True) and len(vitals):
        chart_path = os.path.join(work_dir, f"_chart_{patient_id}.png")
        fig = plot_patient_timeline(patient_id, data_dir=data_dir, layout="separate",
                                     tracks=["signal"], base_fontsize=10)["signal"]
        fig.savefig(chart_path, dpi=150, bbox_inches="tight")
        w_in, h_in = fig.get_size_inches()
        plt.close(fig)
        story.append(Paragraph("Continuous Signal", ss["Section"]))
        story.append(Image(chart_path, width=6.3 * inch, height=6.3 * inch * h_in / w_in))

    if sections.get("labs", True) and len(labs):
        story.append(Paragraph("Labs \u2014 drawn vs. resulted", ss["Section"]))
        rows = [["Test", "Drawn", "Resulted", "Lag", "Flag"]]
        for _, r in labs.sort_values("charttime").iterrows():
            lag_h = (r.storetime - r.charttime).total_seconds() / 3600
            rows.append([r.label, r.charttime.strftime("%d %b %H:%M"), r.storetime.strftime("%d %b %H:%M"),
                         f"{lag_h:.1f}h", r.flag if pd.notna(r.flag) else "normal"])
        story.append(_table(rows, [1.3 * inch, 1.45 * inch, 1.45 * inch, 0.65 * inch, 0.85 * inch], header_bg=C_TEAL))

    if sections.get("notes_timeline", True) and len(notes):
        story.append(Paragraph("Narrative Timeline \u2014 ordered by when it happened", ss["Section"]))
        timeline_notes = notes[notes.note_type != "amendment"] if has_amendments else notes
        amended_ids = set(notes.loc[notes.is_amendment == 1, "amends_note_id"]) if has_amendments else set()
        for _, n in timeline_notes.sort_values("event_charttime").iterrows():
            lag_h = (n.storetime - n.event_charttime).total_seconds() / 3600
            in_window = pd.notna(label.true_onset_time) and (
                label.possible_window_start <= n.event_charttime <= label.possible_window_end)
            header = (f"<b>{n.event_charttime:%d %b %H:%M}</b> happened \u2192 "
                      f"<b>{n.storetime:%d %b %H:%M}</b> charted ({lag_h:.1f}h later) "
                      f"\u2014 {n.note_type.replace('_', ' ')}")
            if n.note_id in amended_ids:
                header += f' <font color="{CORAL}">\u2014 later amended, see Corrections \u2193</font>'
            if in_window:
                header = f'<font color="{CORAL}">\u26a0 in possible event window \u2014 </font>' + header
            story.append(Paragraph(header, ss["Body"]))
            story.append(Paragraph(n.text, ss["NoteBody"]))

    if has_amendments and sections.get("medications", True):
        amendments = notes[notes.is_amendment == 1]
        if len(amendments):
            story.append(Paragraph("Corrections & Amendments", ss["Section"]))
            for _, a in amendments.sort_values("storetime").iterrows():
                orig = notes[notes.note_id == a.amends_note_id]
                exposure_txt = ""
                if len(orig):
                    exposure_h = (a.storetime - orig.iloc[0].storetime).total_seconds() / 3600
                    exposure_txt = (f" The uncorrected version was the record of truth for "
                                     f"{exposure_h:.1f}h before this correction was charted.")
                header = (f"<b>{a.storetime:%d %b %H:%M}</b> \u2014 correction to note "
                          f"<b>{a.amends_note_id}</b> ({a.amendment_reason}).{exposure_txt}")
                story.append(Paragraph(header, ss["Body"]))
                story.append(Paragraph(a.text, ss["NoteBody"]))

    if sections.get("medications", True):
        med_path = f"{data_dir}/synthetic_medication_labels.csv"
        med_notes = notes[notes.note_type != "amendment"] if has_amendments else notes
        if os.path.exists(med_path) and len(med_notes):
            med_labels = pd.read_csv(med_path)
            patient_meds = med_labels[med_labels.note_id.isin(med_notes.note_id)]
            if len(patient_meds):
                story.append(Paragraph("Medications \u2014 extracted from charted text spans", ss["Section"]))
                for note_id, grp in patient_meds.groupby("note_id"):
                    fields = {row.Annotation: row.Text for _, row in grp.iterrows()}
                    line = ", ".join(f"<b>{k}</b>: {v}" for k, v in fields.items())
                    if note_id in amended_ids:
                        line += f' <font color="{CORAL}">\u2014 later amended, see Corrections above</font>'
                    story.append(Paragraph(line, ss["Body"]))

    for track, id_col, time_col, title in [("ecg", "ecg_id", "ecg_time", "ECG Studies"),
                                            ("echo", "echo_id", "echo_time", "Echo Studies")]:
        events = b[track]
        if sections.get(track, True) and len(events):
            story.append(Paragraph(title, ss["Section"]))
            rows = [["Performed", "Report ready", "Interpreted"]]
            for _, e in events.iterrows():
                interp = notes[notes.related_event_id == e[id_col]]
                interpreted = interp.iloc[0].storetime.strftime("%d %b %H:%M") if len(interp) else "\u2014 not yet charted"
                rows.append([e[time_col].strftime("%d %b %H:%M"), e.report_time.strftime("%d %b %H:%M"), interpreted])
            story.append(_table(rows, [1.9 * inch, 1.9 * inch, 1.9 * inch], header_bg=C_TEAL))

    story.append(Spacer(1, 14))
    story.append(Paragraph(
        "Every timestamp above comes directly from the source record. Nothing here was inferred "
        "or backfilled, only reorganised around when things actually happened.", ss["Small"]))

    doc.build(story)
    return save_path
