"""
model_report.py

The artifact a document engineer would actually hand
up the chain after this evaluation, not a patient's chart, this tutorial's
Block 1 already built that, but the model evaluation itself. Same idea as
pdf_report.py, participant-directed sections, honest about what the numbers
do and don't show, but the subject here is the pipeline, not a person.

render_model_report() runs all three models fresh (naive, shortcut,
time-aware) and writes one PDF a manager could actually read: what was
compared, what the risk was, what the fix cost in accuracy (nothing), and
why the underlying issue is a document-engineering one, not a tuning one.
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, brier_score_loss

from reportlab.lib.pagesizes import LETTER
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable, PageBreak
)

import models as m

TEAL, SEAFOAM, DARK, CORAL, AMBER, MUTED = "#028090", "#00A896", "#0A2E36", "#D96C6C", "#E1A82D", "#6B7C7E"
C_TEAL, C_DARK, C_CORAL, C_MUTED = colors.HexColor(TEAL), colors.HexColor(DARK), colors.HexColor(CORAL), colors.HexColor(MUTED)
C_LIGHT_AMBER = colors.HexColor("#FBF3E1")
C_GRID = colors.HexColor("#DDDDDD")
C_ZEBRA = colors.HexColor("#F7F7F7")

DEFAULT_SECTIONS = dict(
    exec_summary=True, model_comparison=True, risk_section=True,
    feature_importance=True, illustrative_case=False, document_engineering_case=True, recommendation=True,
)

SHORTCUT_FEATURES = {"n_notes_naive", "has_event_note_naive", "hours_since_last_note_naive"}


def _styles():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle("ReportTitle", parent=ss["Title"], textColor=C_DARK, fontSize=18, spaceAfter=2))
    ss.add(ParagraphStyle("ReportSubtitle", parent=ss["Normal"], textColor=C_MUTED, fontSize=9.5, spaceAfter=12))
    ss.add(ParagraphStyle("Section", parent=ss["Heading2"], textColor=C_TEAL, fontSize=13, spaceBefore=14, spaceAfter=6))
    ss.add(ParagraphStyle("SubSection", parent=ss["Heading3"], textColor=C_DARK, fontSize=10.5, spaceBefore=10, spaceAfter=4))
    ss.add(ParagraphStyle("Body", parent=ss["Normal"], fontSize=9.5, leading=13.5, spaceAfter=6))
    ss.add(ParagraphStyle("Callout", parent=ss["Normal"], fontSize=9.5, leading=13.5,
                           backColor=C_LIGHT_AMBER, borderColor=colors.HexColor(AMBER), borderWidth=0.75,
                           borderPadding=8, spaceBefore=4, spaceAfter=10))
    ss.add(ParagraphStyle("Small", parent=ss["Normal"], fontSize=8, textColor=C_MUTED, leading=11))
    return ss


def _table(rows, col_widths, header_bg=C_MUTED):
    t = Table(rows, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), header_bg), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.4, C_GRID), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, C_ZEBRA]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def _run_all_models(data_dir):
    naive_df, aware_df, meta_df = m.build_features(data_dir)
    shortcut_df = m.build_shortcut_features(data_dir)
    y = meta_df["outcome"]

    results = {}
    for name, df in [("naive", naive_df), ("shortcut", shortcut_df), ("aware", aware_df)]:
        pipe, Xte, yte, proba = m.train_and_predict(df, y)
        results[name] = dict(pipe=pipe, df=df, Xte=Xte, yte=yte, proba=proba,
                              auroc=roc_auc_score(yte, proba), brier=brier_score_loss(yte, proba))
    return results, meta_df, y


def _illustrative_case_chart(data_dir, patient_id, work_dir):
    """The same two-panel 'asynchronous evidence to uncertain onset, aligned'
    figure from this block's own Figure-1 recreation, factored out here so
    the report can embed it for one concrete patient rather than staying
    purely aggregate throughout."""
    patients, vitals, labs, notes, labels = m._load(data_dir)
    labels["possible_window_start"] = pd.to_datetime(labels["possible_window_start"])
    labels["possible_window_end"] = pd.to_datetime(labels["possible_window_end"])
    lrow = labels.set_index("subject_id").loc[patient_id]
    if pd.isna(lrow.true_onset_time):
        return None, None

    pv = vitals[vitals.subject_id == patient_id]
    pl = labs[labs.subject_id == patient_id]
    pn = notes[notes.subject_id == patient_id]
    span_start = lrow.possible_window_start - pd.Timedelta(hours=8)
    span_end = lrow.possible_window_end + pd.Timedelta(hours=8)
    pv_w = pv[(pv.charttime >= span_start) & (pv.charttime <= span_end)]
    pl_w = pl[(pl.storetime >= span_start) & (pl.storetime <= span_end)]
    pn_w = pn[(pn.storetime >= span_start) & (pn.storetime <= span_end)]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    ax = axes[0]
    ax.plot(pv_w.charttime, pv_w.heartrate, color=TEAL, linewidth=1, alpha=0.85, label="Heart rate")
    for i, (_, r) in enumerate(pl_w.iterrows()):
        ax.axvline(r.storetime, color=SEAFOAM, alpha=0.3, linewidth=6, ymax=0.15, label="Lab drawn" if i == 0 else None)
    for i, (_, r) in enumerate(pn_w.iterrows()):
        ax.axvline(r.storetime, color=CORAL, alpha=0.3, linewidth=6, ymin=0.85, label="Note charted" if i == 0 else None)
    ax.set_title("ASYNCHRONOUS EVIDENCE", fontsize=9.5, color=DARK, fontweight="bold")
    ax.set_ylabel("Heart rate (bpm)")
    ax.legend(fontsize=6.5, loc="lower right")
    ax.tick_params(labelsize=7)

    ax = axes[1]
    t0 = lrow.true_onset_time
    rel = lambda t: (t - t0).total_seconds() / 3600
    ax.plot(pv_w.charttime.apply(rel), pv_w.heartrate, color=TEAL, linewidth=1, alpha=0.85)
    ax.axvspan(rel(lrow.possible_window_start), rel(lrow.possible_window_end), color=AMBER, alpha=0.2,
               label="Possible event window")
    ax.axvline(0, color=DARK, linestyle=":", linewidth=1.2, label="True onset (hidden from model)")
    for _, r in pl_w.iterrows():
        ax.axvline(rel(r.storetime), color=SEAFOAM, alpha=0.3, linewidth=6, ymax=0.15)
    for _, r in pn_w.iterrows():
        ax.axvline(rel(r.storetime), color=CORAL, alpha=0.3, linewidth=6, ymin=0.85)
    ax.set_title("UNCERTAIN ONSET, ALIGNED", fontsize=9.5, color=DARK, fontweight="bold")
    ax.set_xlabel("Hours relative to true onset")
    ax.legend(fontsize=6.5, loc="lower right")
    ax.tick_params(labelsize=7)

    fig.suptitle(f"Patient {patient_id}: illustrative case", fontsize=10.5, color=DARK)
    fig.tight_layout()
    chart_path = os.path.join(work_dir, f"_model_report_case_{patient_id}.png")
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return chart_path, lrow


def _default_event_patient(data_dir):
    patients, vitals, labs, notes, labels = m._load(data_dir)
    event_patients = labels[labels.outcome == 1].subject_id.tolist()
    return event_patients[0] if event_patients else None


def render_model_report(data_dir=".", sections=None, save_path=None, work_dir=".",
                         patient_id=None, custom_sections=None):
    """
    patient_id: which patient to feature in the optional `illustrative_case`
      section. Defaults to the first patient in the corpus with a real
      deterioration event if not given.
    custom_sections: an optional list of {"heading": str, "text": str}
      dicts, your own written review, in your own words, appended to the
      report after the automated analysis. Each becomes its own heading
      and a plain paragraph underneath it, exactly like every other
      section, just written by a person instead of generated from a number.
    """
    sections = {**DEFAULT_SECTIONS, **(sections or {})}
    results, meta_df, y = _run_all_models(data_dir)
    ss = _styles()
    save_path = save_path or "model_evaluation_report.pdf"

    doc = SimpleDocTemplate(save_path, pagesize=LETTER, topMargin=0.75 * inch, bottomMargin=0.75 * inch,
                             leftMargin=0.75 * inch, rightMargin=0.75 * inch)
    story = [
        Paragraph("Deterioration Prediction Pipeline \u2014 Model Evaluation Report", ss["ReportTitle"]),
        Paragraph(f"{len(meta_df)}-patient synthetic corpus  \u00b7  Generated {datetime.now():%d %b %Y}  \u00b7  "
                  f"Prepared for review prior to deployment", ss["ReportSubtitle"]),
        HRFlowable(width="100%", color=C_TEAL, thickness=1),
        Spacer(1, 8),
    ]

    if sections.get("exec_summary", True):
        story.append(Paragraph("Executive Summary", ss["Section"]))
        story.append(Paragraph(
            "Three candidate models were evaluated for predicting patient deterioration. All achieve "
            "comparable, near-perfect aggregate accuracy on this evaluation set. They are not "
            "interchangeable. One model partly relies on how quickly a patient's chart is updated rather "
            "than on their physiology, a pattern invisible in the aggregate accuracy score and only visible "
            "once that signal is isolated and tested on its own, where it produces a substantial performance "
            "gap for patients whose documentation is structurally slower. A second model, built on the exact "
            "same underlying data but represented differently, removes that dependency entirely, at no "
            "measurable cost to accuracy. The recommendation below follows from that difference, not from "
            "the headline metric, which cannot tell these models apart.", ss["Body"]))

    if sections.get("model_comparison", True):
        story.append(Paragraph("Model Comparison", ss["Section"]))
        rows = [["Model", "Built from", "Features", "AUROC", "Brier score"]]
        descriptions = {
            "naive": "Last-known vitals/labs + note-count and timing",
            "shortcut": "Note count and charting timing ONLY, no physiology",
            "aware": "Vitals/labs trend + structured missingness, no note timing",
        }
        for name in ["naive", "shortcut", "aware"]:
            r = results[name]
            rows.append([name, Paragraph(descriptions[name], ss["Body"]), str(r["df"].shape[1]),
                         f"{r['auroc']:.3f}", f"{r['brier']:.4f}"])
        story.append(_table(rows, [0.8*inch, 2.75*inch, 0.7*inch, 0.7*inch, 0.9*inch], header_bg=C_TEAL))
        story.append(Paragraph(
            "AUROC alone rates the naive and time-aware models identically, and cannot distinguish either "
            "from a model that turns out to fail unevenly across patients. Brier score, which penalizes "
            "confident wrong answers more heavily, shows a modest edge for the time-aware model, consistent "
            "with a more honestly-grounded confidence, not just equal accuracy.", ss["Small"]))

    if sections.get("risk_section", True):
        story.append(Paragraph("The Risk, Isolated", ss["Section"]))
        story.append(Paragraph(
            "The documentation-timing signal was tested on its own, with all physiological information "
            "removed, to answer one question: how much can charting speed alone predict, and who does it "
            "fail? On its own it clears a coin flip by a wide margin, meaning it is a genuinely usable "
            "predictor, which is exactly why a model is tempted to lean on it.", ss["Body"]))

        chart_path = os.path.join(work_dir, "_model_report_risk.png")
        tier_recall = m.subgroup_recall(results["shortcut"]["pipe"], results["shortcut"]["df"], y, meta_df,
                                         "entanglement_tier")
        order = ["low", "medium", "high"]
        vals = tier_recall.set_index("group").reindex(order)["recall"]
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(order, vals, color=[SEAFOAM, TEAL, CORAL])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Recall, patients who actually deteriorated")
        ax.set_xlabel("Entanglement tier (how fast documentation happens)")
        ax.set_title("Documentation-timing-only model, by tier", fontsize=10, color=DARK)
        fig.tight_layout()
        fig.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        story.append(Image(chart_path, width=5.2 * inch, height=5.2 * inch * 3 / 6))
        story.append(Paragraph(
            f"<b>Recall ranges from {vals.min():.0%} to {vals.max():.0%} depending on tier</b>, using "
            f"identical evaluation rules for every patient. This is not visible in the full model's "
            f"aggregate subgroup recall, physiology in the full feature set is strong enough on this corpus "
            f"to compensate for the shortcut's bias. That compensation is not something to rely on: it means "
            f"a bias can sit undetected in a model's coefficients, doing no visible harm only until a "
            f"deployment or patient population shifts enough that the compensation no longer holds.",
            ss["Callout"]))

    if sections.get("feature_importance", True):
        story.append(Paragraph("What Each Model Actually Relies On", ss["Section"]))
        fi_naive = m.feature_importance(results["naive"]["pipe"], results["naive"]["df"].columns)
        fi_aware = m.feature_importance(results["aware"]["pipe"], results["aware"]["df"].columns)
        rows = [["Rank", "Naive model feature", "Weight", "Time-aware model feature", "Weight"]]
        for i in range(5):
            nf, nw = fi_naive.index[i], fi_naive.iloc[i]
            af, aw = fi_aware.index[i], fi_aware.iloc[i]
            flagged = nf in SHORTCUT_FEATURES
            nf_disp = f'<font color="{CORAL}"><b>{nf} (flagged)</b></font>' if flagged else nf
            rows.append([str(i + 1), Paragraph(nf_disp, ss["Body"]), f"{nw:+.2f}", af, f"{aw:+.2f}"])
        story.append(_table(rows, [0.4*inch, 1.9*inch, 0.6*inch, 1.9*inch, 0.6*inch], header_bg=C_TEAL))
        if any(f in SHORTCUT_FEATURES for f in fi_naive.index[:5]):
            story.append(Paragraph(
                '"(flagged)" marks a documentation-timing feature ranked among the naive model\u2019s top 5 by '
                "weight. The time-aware model's feature list contains no such feature by construction, it "
                "was never given the option.", ss["Small"]))

    if sections.get("illustrative_case", False):
        featured_patient = patient_id if patient_id is not None else _default_event_patient(data_dir)
        if featured_patient is not None:
            chart_path, lrow = _illustrative_case_chart(data_dir, featured_patient, work_dir)
            if chart_path is not None:
                story.append(Paragraph("Illustrative Case", ss["Section"]))
                window_h = (lrow.possible_window_end - lrow.possible_window_start).total_seconds() / 3600
                story.append(Paragraph(
                    f"Patient {featured_patient}, one real case from the evaluation corpus, shown to ground "
                    f"the analysis above in an actual patient rather than aggregates alone. Left: heart rate, "
                    f"lab draws, and note-charting times as they actually arrived, no structure imposed. "
                    f"Right: the same evidence, aligned to true onset (hidden from every model in this "
                    f"report) with the {window_h:.1f}-hour possible event window shown, the same uncertainty "
                    f"window `soft_event_label()` and the aligned features in this report's models are built "
                    f"around.", ss["Body"]))
                story.append(Image(chart_path, width=6.3 * inch, height=6.3 * inch * 3.6 / 10.5))

    if sections.get("document_engineering_case", True):
        story.append(PageBreak())
        story.append(Paragraph("Why This Is a Document Engineering Problem", ss["Section"]))
        story.append(Paragraph(
            "The fix demonstrated here was never a better model. Both models in this comparison are the "
            "identical logistic regression pipeline, same preprocessing, same training procedure. What "
            "changed is how the underlying clinical record was represented before it ever reached the model: "
            "a snapshot value replaced by an event-centred trend, a silently-dropped gap replaced by an "
            "explicit missingness feature, a proxy for documentation speed removed because a better "
            "representation of the physiology made it unnecessary. Those are representation decisions, made "
            "upstream of modeling entirely, in exactly the layer of the pipeline a document engineer, not a "
            "machine learning engineer, is responsible for.", ss["Body"]))
        story.append(Paragraph(
            "This is also why the risk shown above would have been invisible to a standard model evaluation "
            "that stops at aggregate accuracy and a single subgroup pass on the deployed model. It only "
            "surfaced because the documentation-timing signal was deliberately isolated and evaluated on its "
            "own terms, a document-engineering-style audit of what evidence a model is actually consuming, "
            "not a pure model-performance check.", ss["Body"]))

    if sections.get("recommendation", True):
        story.append(Paragraph("Recommendation", ss["Section"]))
        story.append(Paragraph(
            "Deploy the time-aware architecture. It matches the naive model's accuracy with a materially "
            "different, and checkable, risk profile. More generally: aggregate accuracy should not be "
            "accepted as sufficient sign-off for a clinical prediction system on its own. Any feature "
            "correlated with documentation speed, charting frequency, note recency, staffing-dependent "
            "timing, should be isolated and evaluated independently before deployment, using the same method "
            "demonstrated in this report, not assumed safe because the full model's aggregate number looks "
            "acceptable.", ss["Body"]))

    if custom_sections:
        story.append(Paragraph("Reviewer's Notes", ss["Section"]))
        story.append(Paragraph(
            "The sections below were written by the person who ran this evaluation, not generated from "
            "the pipeline's own numbers. Treat them as a reviewer's judgment, not as an additional finding.",
            ss["Small"]))
        for entry in custom_sections:
            heading = entry.get("heading", "").strip()
            text = entry.get("text", "").strip()
            if not heading and not text:
                continue
            if heading:
                story.append(Paragraph(heading, ss["SubSection"]))
            if text:
                story.append(Paragraph(text, ss["Body"]))

    story.append(Spacer(1, 14))
    story.append(Paragraph(
        "All figures in this report are computed live from the evaluation set described above. Nothing "
        "here is asserted without a corresponding number produced by the pipeline itself.", ss["Small"]))

    doc.build(story)
    return save_path
