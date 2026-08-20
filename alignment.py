"""
alignment.py

Block 2 alignment utilities for the DocEng '26 tutorial.

Everything here answers one question: given a prediction time T for a
patient, what should "the data we have so far" actually mean? There are two
very different answers, and the gap between them is the whole point of this
tutorial.

  NAIVE approach (what most pipelines do by default):
    Pool everything with a timestamp <= T into flat aggregates. A note
    counts as "available" the moment it's charted. Blunt features like
    "how many notes mention concern" get treated the same as a lab value.

  TIME-AWARE approach:
    Treat the continuous/intermittent streams as trend evidence (what IS
    the physiology doing, not just what's the latest single number), treat
    narrative evidence as uncertain rather than as a hard yes/no proxy, and
    make missingness itself a visible feature rather than something to
    silently drop.

Neither approach can see into the future, both only ever use storetime <= T
for anything that has to be "charted" before it exists. The naive approach's
mistake isn't time travel. It's what it leans on.
"""

import numpy as np
import pandas as pd


# Windowing

def naive_window(df, time_col, prediction_time, lookback_hours=48):
    """Everything with time_col in (prediction_time - lookback, prediction_time].
    No distinction between 'when it happened' and 'when it was charted', this
    is the fixed-window approach used as-is throughout most naive pipelines."""
    lo = prediction_time - pd.Timedelta(hours=lookback_hours)
    return df[(df[time_col] > lo) & (df[time_col] <= prediction_time)]


def event_centered_trend(df, time_col, value_col, prediction_time, lookback_hours=48):
    """Returns (last_value, trend_slope, n_observations) for a numeric stream,
    computed only from observations properly timestamped at or before
    prediction_time. The trend slope is what makes this 'event-centred'
    rather than a single blind snapshot: a value that's rising matters
    differently than the same value sitting flat."""
    window = naive_window(df, time_col, prediction_time, lookback_hours)
    window = window.dropna(subset=[value_col])
    if len(window) == 0:
        return np.nan, np.nan, 0
    if len(window) == 1:
        return float(window[value_col].iloc[0]), 0.0, 1

    t0 = window[time_col].min()
    hours = (window[time_col] - t0).dt.total_seconds() / 3600
    slope = np.polyfit(hours, window[value_col], 1)[0]
    last_value = float(window.sort_values(time_col)[value_col].iloc[-1])
    return last_value, float(slope), len(window)


# Soft / probabilistic labelling under onset uncertainty
  
def soft_event_label(evaluation_time, window_start, window_end):
    """A patient's deterioration doesn't switch on at a single instant, it's
    somewhere in an uncertainty window. Rather than a hard 0/1 label at an
    arbitrary cut, this returns a soft probability: 0 before the window,
    ramping linearly across it, 1 after. Useful for training-time label
    smoothing, distinct from the hard 'outcome' column used for evaluation."""
    if pd.isna(window_start) or pd.isna(window_end):
        return 0.0
    if evaluation_time <= window_start:
        return 0.0
    if evaluation_time >= window_end:
        return 1.0
    return (evaluation_time - window_start) / (window_end - window_start)


# Structured missingness
  
def missingness_features(df, time_col, prediction_time, lookback_hours=48, expected_per_day=None):
    """How much evidence is actually present in the window, relative to how
    much we'd expect, is itself informative (Part 4: missingness is not
    random). Returns (n_observed, expected_gap_ratio)."""
    window = naive_window(df, time_col, prediction_time, lookback_hours)
    n = len(window)
    if expected_per_day is None or n == 0:
        return n, np.nan
    expected = expected_per_day * (lookback_hours / 24)
    return n, n / max(expected, 1e-6)


# Note-derived signal, built two different ways on purpose
  
def naive_note_signal(notes_df, prediction_time, lookback_hours=48):
    """The shortcut: how many notes exist, and is one of them 'concerning',
    filtered purely by storetime. This is exactly the kind of easy,
    high-signal-looking feature a naive pipeline reaches for, and exactly
    the one that quietly encodes documentation speed rather than physiology.

    Amendment rows (note_type == "amendment", added by generate_amendments.py
    for Block 1) are excluded here on purpose: an amendment is a correction
    to an existing note, not a new piece of charting activity, and counting
    it as one would silently change n_notes_naive / hours_since_last_note_naive
    every time the data is regenerated, for reasons that have nothing to do
    with what this block is teaching. If you regenerate this function from
    scratch, keep this filter."""
    notes_df = notes_df[notes_df.note_type != "amendment"] if "note_type" in notes_df.columns else notes_df
    window = naive_window(notes_df, "storetime", prediction_time, lookback_hours)
    n_notes = len(window)
    has_event_note = int((window.note_type.isin(["event_clinician", "event_overshadowed"])).any())
    if n_notes == 0:
        hours_since_last = lookback_hours
    else:
        hours_since_last = (prediction_time - window.storetime.max()).total_seconds() / 3600
    return dict(n_notes_naive=n_notes, has_event_note_naive=has_event_note,
                hours_since_last_note_naive=hours_since_last)
