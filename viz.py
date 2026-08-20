"""
viz.py

Three plots:

  plot_patient_timeline(patient_id): the Figure-1-style multi-track view.
      Continuous signal, intermittent labs, delayed notes, and (when present)
      ECG and echo, each on their own track, all sharing one time axis. This
      is the plot that makes asynchrony visible rather than asserted.

      layout="stacked" (default): one tall figure, tracks stacked vertically.
      layout="separate": one self-contained figure per track, returned as a
          dict {track_name: fig}. Useful for slides, or when the stacked
          text is too small to read comfortably.
      layout="grid": all tracks arranged in an `ncols`-wide grid, still
          sharing one time axis.
      `tracks` lets you pick a subset, e.g. tracks=["signal", "notes"].
      `base_fontsize` scales every label/legend/tick in the figure at once.

  plot_event_window(patient_id): the "possible event window" diagram.
      True onset plus the uncertainty band around it, with the first
      documented note overlaid, so the gap between "when it happened" and
      "when someone wrote it down" has a visible shape.

  plot_lag_by_tier(): the aggregate calibration payoff.
      Documentation lag distribution across all patients, split by
      entanglement tier, computed on the fly from raw timestamps (never
      precomputed in the CSVs, that's the point).

"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D


# Palette (kept consistent with the tutorial deck: Teal Trust)

TEAL = "#028090"
SEAFOAM = "#00A896"
MINT = "#02C39A"
DARK = "#0A2E36"
MUTED = "#6B7C7E"
CARD = "#EAF5F4"
AMBER = "#E1A82D"
CORAL = "#D96C6C"

NOTE_STYLE = {
    "admission":           dict(marker="o", color=MUTED,   label="Admission note"),
    "routine_stable":      dict(marker="o", color=SEAFOAM, label="Routine note"),
    "event_clinician":     dict(marker="^", color=AMBER,   label="Event note (clinician)"),
    "event_overshadowed":  dict(marker="^", color=CORAL,   label="Event note (overshadowed)"),
    "caregiver":           dict(marker="D", color=TEAL,    label="Caregiver note"),
    "discharge":           dict(marker="s", color=MUTED,   label="Discharge note"),
    "ecg_interpretation":  dict(marker="P", color=DARK,    label="ECG interpretation"),
    "echo_interpretation": dict(marker="X", color=DARK,    label="Echo interpretation"),
    "amendment":           dict(marker="*", color=CORAL,   label="Amendment (correction to a prior note)"),
}

# Relative height each track gets in "stacked" layout, and a rough per-panel
# height (inches) used to size "separate" and "grid" figures.
TRACK_HEIGHT_RATIOS = {"signal": 2.0, "labs": 1.5, "notes": 1.8, "ecg": 1.3, "echo": 1.3}
TRACK_TITLES = {
    "signal": "Continuous signal (heart rate, O2 sat)",
    "labs": "Intermittent measurements (labs)",
    "notes": "Delayed narrative (clinical notes)",
    "ecg": "ECG \u2014 performed \u2192 technical report \u2192 clinician read",
    "echo": "Echo \u2014 performed \u2192 technical report \u2192 clinician read",
}


def load_patient_bundle(patient_id, data_dir="."):
    """Load every stream for one patient into a dict of DataFrames."""
    p = pd.read_csv(f"{data_dir}/synthetic_patients.csv", parse_dates=["admittime", "dischtime"])
    patient = p[p.subject_id == patient_id]
    if patient.empty:
        raise ValueError(f"subject_id {patient_id} not found in synthetic_patients.csv")
    patient = patient.iloc[0]

    vitals = pd.read_csv(f"{data_dir}/synthetic_vitals_continuous.csv", parse_dates=["charttime"])
    vitals = vitals[vitals.subject_id == patient_id].sort_values("charttime")

    labs = pd.read_csv(f"{data_dir}/synthetic_labs_intermittent.csv", parse_dates=["charttime", "storetime"])
    labs = labs[labs.subject_id == patient_id].sort_values("charttime")

    notes = pd.read_csv(f"{data_dir}/synthetic_notes.csv", parse_dates=["event_charttime", "storetime"])
    notes = notes[notes.subject_id == patient_id].sort_values("storetime")

    ecg = pd.read_csv(f"{data_dir}/synthetic_ecg_events.csv", parse_dates=["ecg_time", "report_time"])
    ecg = ecg[ecg.subject_id == patient_id].sort_values("ecg_time")

    echo = pd.read_csv(f"{data_dir}/synthetic_echo_events.csv", parse_dates=["echo_time", "report_time"])
    echo = echo[echo.subject_id == patient_id].sort_values("echo_time")

    labels = pd.read_csv(f"{data_dir}/synthetic_labels.csv",
                          parse_dates=["true_onset_time", "possible_window_start", "possible_window_end"])
    label = labels[labels.subject_id == patient_id].iloc[0]

    return dict(patient=patient, vitals=vitals, labs=labs, notes=notes, ecg=ecg, echo=echo, label=label)


def _fontsizes(base):
    """Every size used across the timeline plot, derived from one number so
    `base_fontsize=` scales the whole figure consistently."""
    return dict(
        suptitle=base + 3,
        track_title=base,
        label=base,
        tick=max(base - 3, 7),
        legend=max(base - 3, 7),
        annotation=max(base - 2, 8),
    )


def _no_data(ax, message, fs):
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes,
            color=MUTED, fontsize=fs["label"], style="italic")
    ax.set_yticks([])


def _draw_onset_context(ax, label):
    """Shade the possible-event window and mark true onset. Drawn on every
    track so the reference point survives being split into separate figures."""
    if pd.notna(label.true_onset_time):
        ax.axvspan(label.possible_window_start, label.possible_window_end,
                   color=AMBER, alpha=0.12, zorder=0)
        ax.axvline(label.true_onset_time, color=AMBER, linestyle="--", linewidth=1.4, zorder=1)


# Track drawers
# Kept separate so any layout (stacked / separate / grid) can call the same
# per-track logic.

def _draw_signal(ax, b, fs):
    vitals = b["vitals"]
    if len(vitals):
        ax.plot(vitals.charttime, vitals.heartrate, color=TEAL, linewidth=1.4, label="Heart rate")
        ax.plot(vitals.charttime, vitals.o2sat, color=SEAFOAM, linewidth=1.4, label="O2 sat", alpha=0.85)
        missing = vitals[vitals.signal_quality == "degraded"]
        if len(missing):
            floor = np.nanmin(np.concatenate([vitals.heartrate.values, vitals.o2sat.values]))
            ax.scatter(missing.charttime, [floor] * len(missing), marker="x", color=CORAL,
                       s=26, zorder=3, label="Degraded reading")
        ax.legend(loc="upper left", fontsize=fs["legend"], framealpha=0.9)
    else:
        _no_data(ax, "No continuous signal data for this patient", fs)
    ax.set_ylabel("Continuous\nsignal", fontsize=fs["label"])
    ax.set_title(TRACK_TITLES["signal"], fontsize=fs["track_title"], loc="left", color=DARK, fontweight="bold")


def _draw_labs(ax, b, fs):
    labs = b["labs"]
    if len(labs):
        colors = [CORAL if f == "abnormal" else SEAFOAM for f in labs.flag.fillna("normal")]
        ax.scatter(labs.charttime, labs.label, c=colors, s=38, zorder=3)
        ax.scatter(labs.storetime, labs.label, c=colors, s=14, marker="|", zorder=2, alpha=0.6)
        for _, r in labs.iterrows():
            ax.plot([r.charttime, r.storetime], [r.label, r.label], color=MUTED, linewidth=0.7, alpha=0.5, zorder=1)
        legend = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor=SEAFOAM, markersize=8, label="Drawn, normal result"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=CORAL, markersize=8, label="Drawn, abnormal result"),
            Line2D([0], [0], marker="|", color=MUTED, markersize=10, linestyle="None", label="Resulted (storetime)"),
        ]
        ax.legend(handles=legend, loc="lower right", fontsize=fs["legend"], framealpha=0.9, ncol=1)
    else:
        _no_data(ax, "No lab draws for this patient", fs)
    ax.set_ylabel("Intermittent\nmeasurements", fontsize=fs["label"])
    ax.set_title(TRACK_TITLES["labs"], fontsize=fs["track_title"], loc="left", color=DARK, fontweight="bold")


def _draw_notes(ax, b, fs):
    notes = b["notes"]
    if len(notes):
        y_positions = {}
        note_id_pos = {}  # note_id -> (y, storetime), used below to connect amendments to what they amend
        for _, n in notes.iterrows():
            style = NOTE_STYLE.get(n.note_type, dict(marker="o", color=MUTED, label=n.note_type))
            y = y_positions.setdefault(n.note_type, len(y_positions))
            ax.plot([n.event_charttime, n.storetime], [y, y], color=style["color"], linewidth=1.1, alpha=0.5, zorder=1)
            ax.scatter(n.event_charttime, y, color=style["color"], s=20, marker="o", facecolors="none", zorder=2)
            ax.scatter(n.storetime, y, color=style["color"], s=50, marker=style["marker"], zorder=3)
            note_id_pos[n.note_id] = (y, n.storetime)

        has_amendments = "is_amendment" in notes.columns and "amends_note_id" in notes.columns
        if has_amendments:
            for _, a in notes[notes.is_amendment == 1].iterrows():
                target = note_id_pos.get(a.amends_note_id)
                if target is None:
                    continue
                target_y, target_storetime = target
                amend_y, _ = note_id_pos[a.note_id]
                ax.plot([target_storetime, a.storetime], [target_y, amend_y], color=CORAL,
                        linewidth=1.0, alpha=0.6, linestyle=":", zorder=1)

        ax.set_yticks(list(y_positions.values()))
        ax.set_yticklabels([t.replace("_", " ") for t in y_positions.keys()], fontsize=fs["tick"])
        ax.set_ylim(-1, max(len(y_positions), 1) + 1.5)  # headroom so the legend clears the top row
        legend = [
            Line2D([0], [0], marker="o", color=MUTED, markerfacecolor="none", markersize=7, linestyle="None",
                   label="Event time (event_charttime)"),
            Line2D([0], [0], marker="o", color=MUTED, markerfacecolor=MUTED, markersize=8, linestyle="None",
                   label="Charted time (storetime)"),
            Line2D([0], [0], color=MUTED, linewidth=1.1, alpha=0.6, label="The gap between them = lag"),
        ]
        if has_amendments and notes.is_amendment.sum() > 0:
            legend.append(Line2D([0], [0], color=CORAL, linewidth=1.0, alpha=0.6, linestyle=":",
                                  label="Amendment \u2192 note it corrects"))
        ax.legend(handles=legend, loc="upper left", fontsize=fs["legend"], framealpha=0.9, ncol=1)
    else:
        _no_data(ax, "No notes for this patient", fs)
    ax.set_ylabel("Delayed\nnarrative", fontsize=fs["label"])
    ax.set_title(TRACK_TITLES["notes"], fontsize=fs["track_title"], loc="left", color=DARK, fontweight="bold")


def _draw_diagnostic_chain(ax, events, notes, time_col, id_col, color, fs, track_title, ylabel):
    """Shared drawing logic for the ECG and echo tracks: performed -> report
    -> clinician read, one row per study."""
    if not len(events):
        _no_data(ax, f"No {ylabel.lower()} studies for this patient", fs)
        ax.set_ylabel(ylabel, fontsize=fs["label"])
        ax.set_title(track_title, fontsize=fs["track_title"], loc="left", color=DARK, fontweight="bold")
        return

    yticks, yticklabels = [], []
    for y, (_, e) in enumerate(events.iterrows()):
        ax.plot([e[time_col], e.report_time], [y, y], color=color, linewidth=1.2, alpha=0.5)
        ax.scatter(e[time_col], y, color=color, s=28, marker="o", facecolors="none", zorder=3)
        ax.scatter(e.report_time, y, color=color, s=38, marker="s", zorder=3)
        interp = notes[notes.related_event_id == e[id_col]]
        if len(interp):
            it = interp.iloc[0].storetime
            ax.plot([e.report_time, it], [y, y], color=color, linewidth=1.2, alpha=0.5, linestyle=":")
            ax.scatter(it, y, color=color, s=58, marker="P", zorder=3)
        yticks.append(y)
        yticklabels.append(e[time_col].strftime("%b %d %H:%M"))

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=fs["tick"])
    ax.set_ylim(-1, max(len(yticks), 1) + 1.5)  # headroom so the legend clears the top row
    ax.set_ylabel(ylabel, fontsize=fs["label"])
    ax.set_title(track_title, fontsize=fs["track_title"], loc="left", color=DARK, fontweight="bold")
    legend = [
        Line2D([0], [0], marker="o", color=MUTED, markerfacecolor="none", markersize=7, linestyle="None", label="Performed"),
        Line2D([0], [0], marker="s", color=MUTED, markerfacecolor=MUTED, markersize=7, linestyle="None", label="Technical report ready"),
        Line2D([0], [0], marker="P", color=MUTED, markerfacecolor=MUTED, markersize=8, linestyle="None", label="Clinician's interpretation charted"),
    ]
    ax.legend(handles=legend, loc="upper left", fontsize=fs["legend"], framealpha=0.9, ncol=1)


def _draw_ecg(ax, b, fs):
    _draw_diagnostic_chain(ax, b["ecg"], b["notes"], "ecg_time", "ecg_id", DARK, fs,
                            TRACK_TITLES["ecg"], "ECG studies")


def _draw_echo(ax, b, fs):
    _draw_diagnostic_chain(ax, b["echo"], b["notes"], "echo_time", "echo_id", TEAL, fs,
                            TRACK_TITLES["echo"], "Echo studies")


TRACK_DRAWERS = {"signal": _draw_signal, "labs": _draw_labs, "notes": _draw_notes,
                  "ecg": _draw_ecg, "echo": _draw_echo}


def _patient_subtitle(patient_id, patient):
    return (f"Patient {patient_id}  |  {patient.hospital_complexity_tier}, "
            f"{patient.entanglement_tier} entanglement, {patient.population_context}, "
            f"{patient.socioeconomic_context}")


def _format_time_axis(ax, fs, minticks=4, maxticks=8, xlabel=True):
    # AutoDateLocator has a known edge case where certain multi-day spans
    # raise "unable to pick an appropriate interval" (a real matplotlib
    # UserWarning, not a false alarm), so pick explicitly based on the
    # actual span rather than let it guess.
    lo, hi = ax.get_xlim()
    span_hours = (hi - lo) * 24
    if span_hours <= 30:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, int(span_hours / 6))))
    elif span_hours <= 24 * 10:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, int(span_hours / 24 / 6))))
    else:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=minticks, maxticks=maxticks))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    ax.tick_params(axis="x", labelsize=fs["tick"])
    if xlabel:
        ax.set_xlabel("Time", fontsize=fs["label"])


# Plot 1: the multi-track timeline

def plot_patient_timeline(patient_id, data_dir=".", layout="stacked", tracks=None,
                           figsize=None, base_fontsize=12, ncols=2, save_path=None, dpi=130):
    """
    layout:
      "stacked"  (default) - one figure, tracks stacked vertically, sharing one time axis.
      "separate"           - one self-contained figure per track. Returns a dict
                              {track_name: Figure} instead of a single Figure.
      "grid"                - all tracks arranged in an `ncols`-wide grid, still
                              sharing one time axis.
    tracks: which tracks to include and in what order, e.g. ["signal", "notes"].
      Defaults to signal + labs + notes, plus ecg/echo if this patient has any.
    base_fontsize: scales every label, tick, and legend in the figure at once.
    save_path: for "stacked"/"grid", the file to write. For "separate", used as
      a base name, e.g. save_path="timeline.png" writes "timeline_signal.png", etc.
    """
    if layout not in ("stacked", "separate", "grid"):
        raise ValueError(f"layout must be 'stacked', 'separate', or 'grid', got {layout!r}")

    b = load_patient_bundle(patient_id, data_dir)
    patient, label = b["patient"], b["label"]
    fs = _fontsizes(base_fontsize)

    if tracks is None:
        tracks = ["signal", "labs", "notes"]
        if len(b["ecg"]):
            tracks.append("ecg")
        if len(b["echo"]):
            tracks.append("echo")
    unknown = set(tracks) - set(TRACK_DRAWERS)
    if unknown:
        raise ValueError(f"Unknown track(s) {unknown}, choose from {list(TRACK_DRAWERS)}")

    subtitle = _patient_subtitle(patient_id, patient)

    # -- layout: separate --------------------------------------------------
    if layout == "separate":
        figs = {}
        for name in tracks:
            h = TRACK_HEIGHT_RATIOS.get(name, 1.5)
            this_figsize = figsize or (13, 2.6 * h)
            fig, ax = plt.subplots(figsize=this_figsize)
            _draw_onset_context(ax, label)
            TRACK_DRAWERS[name](ax, b, fs)
            _format_time_axis(ax, fs)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            fig.suptitle(subtitle, fontsize=fs["suptitle"], x=0.01, ha="left", color=DARK, fontweight="bold")
            fig.tight_layout(rect=[0, 0, 1, 0.90])
            if save_path:
                root, ext = os.path.splitext(save_path)
                fig.savefig(f"{root}_{name}{ext or '.png'}", dpi=dpi, bbox_inches="tight")
            figs[name] = fig
        return figs

    # layout: grid 
    if layout == "grid":
        n = len(tracks)
        nrows = int(np.ceil(n / ncols))
        this_figsize = figsize or (7.5 * ncols, 3.6 * nrows)
        fig, axes = plt.subplots(nrows, ncols, figsize=this_figsize, sharex=True, squeeze=False)
        axes = axes.flatten()

        # bottom-most active row in each column gets the date axis
        col_last_row = {}
        for i in range(n):
            r, c = divmod(i, ncols)
            col_last_row[c] = max(col_last_row.get(c, -1), r)

        for i, name in enumerate(tracks):
            ax = axes[i]
            r, c = divmod(i, ncols)
            _draw_onset_context(ax, label)
            TRACK_DRAWERS[name](ax, b, fs)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if r == col_last_row[c]:
                _format_time_axis(ax, fs)
            else:
                ax.tick_params(axis="x", labelbottom=False)
        for j in range(n, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(subtitle, fontsize=fs["suptitle"], x=0.01, ha="left", color=DARK, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    #layout: stacked (default) 
    n = len(tracks)
    ratios = [TRACK_HEIGHT_RATIOS.get(t, 1.5) for t in tracks]
    this_figsize = figsize or (14, 2.3 * sum(ratios))
    fig, axes = plt.subplots(n, 1, figsize=this_figsize, sharex=True,
                              gridspec_kw=dict(height_ratios=ratios))
    axes = np.atleast_1d(axes)

    for ax, name in zip(axes, tracks):
        _draw_onset_context(ax, label)
        TRACK_DRAWERS[name](ax, b, fs)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    _format_time_axis(axes[-1], fs)
    fig.suptitle(subtitle, fontsize=fs["suptitle"], x=0.01, ha="left", color=DARK, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


# Plot 2: the event-window diagram

def plot_event_window(patient_id, data_dir=".", figsize=(10, 3.5), base_fontsize=12, save_path=None):
    b = load_patient_bundle(patient_id, data_dir)
    label, notes = b["label"], b["notes"]
    fs = _fontsizes(base_fontsize)

    fig, ax = plt.subplots(figsize=figsize)

    if pd.isna(label.true_onset_time):
        ax.text(0.5, 0.5, "This patient had no deterioration event (outcome = 0).",
                ha="center", va="center", fontsize=fs["label"], color=MUTED, transform=ax.transAxes)
        ax.set_axis_off()
        if save_path:
            fig.savefig(save_path, dpi=130, bbox_inches="tight")
        return fig

    ax.axvspan(label.possible_window_start, label.possible_window_end, color=AMBER, alpha=0.25,
               label=f"Possible event window ({(label.possible_window_end - label.possible_window_start).total_seconds()/3600:.1f}h wide)")
    ax.axvline(label.true_onset_time, color=DARK, linewidth=1.6, linestyle="--", label="True onset (answer key)")

    event_notes = notes[notes.note_type.isin(["event_clinician", "event_overshadowed"])]
    if len(event_notes):
        first = event_notes.sort_values("storetime").iloc[0]
        ax.axvline(first.storetime, color=CORAL, linewidth=1.6, label="First documented (clinician note)")
        lag_h = (first.storetime - label.true_onset_time).total_seconds() / 3600
        ax.annotate(f"documentation lag: {lag_h:.1f}h", xy=(first.storetime, 0.7), xycoords=("data", "axes fraction"),
                    fontsize=fs["annotation"], color=CORAL, ha="left")

    ax.set_yticks([])
    _format_time_axis(ax, fs, minticks=4, maxticks=7, xlabel=False)
    ax.legend(loc="upper left", fontsize=fs["legend"], framealpha=0.9)
    ax.set_title(f"Patient {patient_id}: uncertain onset vs. documentation lag", fontsize=fs["track_title"],
                 loc="left", color=DARK, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
    return fig


# Plot 3: aggregate documentation lag by entanglement tier

def plot_lag_by_tier(data_dir=".", figsize=(8, 5.5), base_fontsize=12, save_path=None):
    patients = pd.read_csv(f"{data_dir}/synthetic_patients.csv")
    notes = pd.read_csv(f"{data_dir}/synthetic_notes.csv", parse_dates=["event_charttime", "storetime"])
    fs = _fontsizes(base_fontsize)

    notes = notes.merge(patients[["subject_id", "entanglement_tier"]], on="subject_id")
    notes["lag_hours"] = (notes.storetime - notes.event_charttime).dt.total_seconds() / 3600

    order = ["low", "medium", "high"]
    data = [notes[notes.entanglement_tier == t].lag_hours.dropna().values for t in order]

    fig, ax = plt.subplots(figsize=figsize)
    # matplotlib >=3.9 renamed boxplot's `labels` kwarg to `tick_labels`
    # (and removed `labels` entirely in later releases). Try the new name
    # first, fall back to the old one so this works across versions.
    try:
        bp = ax.boxplot(data, tick_labels=order, patch_artist=True, showfliers=False)
    except TypeError:
        bp = ax.boxplot(data, labels=order, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], [SEAFOAM, TEAL, CORAL]):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.tick_params(axis="both", labelsize=fs["tick"])
    ax.set_ylabel("Documentation lag (hours)\n(computed live: storetime \u2212 event_charttime)", fontsize=fs["label"])
    ax.set_xlabel("Entanglement tier", fontsize=fs["label"])
    ax.set_title("Documentation lag by entanglement tier, computed from raw notes.csv",
                 fontsize=fs["track_title"], color=DARK, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
    return fig


# Demo

if __name__ == "__main__":
    import os
    DATA_DIR = "."
    OUT = "viz_demo_output"
    os.makedirs(OUT, exist_ok=True)

    patients = pd.read_csv(f"{DATA_DIR}/synthetic_patients.csv")
    labels = pd.read_csv(f"{DATA_DIR}/synthetic_labels.csv")
    ecg = pd.read_csv(f"{DATA_DIR}/synthetic_ecg_events.csv")

    # pick three illustrative patients: one dementia_severe with an event,
    # one clean low-entanglement, one with ECG/echo activity
    dem = labels[(labels.population_context == "dementia_severe") & (labels.outcome == 1)]
    clean = labels[(labels.entanglement_tier == "low") & (labels.population_context == "none")]
    with_ecg = patients[patients.subject_id.isin(ecg.subject_id)]

    picks = []
    if len(dem): picks.append(("dementia_severe_event", int(dem.iloc[0].subject_id)))
    if len(clean): picks.append(("clean_low_entanglement", int(clean.iloc[0].subject_id)))
    if len(with_ecg): picks.append(("has_ecg_echo", int(with_ecg.iloc[0].subject_id)))

    for tag, pid in picks:
        print("plotting", tag, pid)
        plot_patient_timeline(pid, DATA_DIR, save_path=f"{OUT}/timeline_{tag}_{pid}.png")
        plt.close("all")
        plot_event_window(pid, DATA_DIR, save_path=f"{OUT}/window_{tag}_{pid}.png")
        plt.close("all")

    plot_lag_by_tier(DATA_DIR, save_path=f"{OUT}/lag_by_tier.png")
    plt.close("all")
    print("done, wrote to", OUT)
