"""
generate_education_domain.py

A small, second synthetic corpus, deliberately not healthcare, built to
prove one specific claim: alignment.py's functions (naive_window,
event_centered_trend, soft_event_label, missingness_features) are genuinely
generic. Nothing about them is imported, wrapped, or modified for this
domain, they're called completely unmodified in the Block 3 notebook,
against this data instead of ICU data.

Same taxonomy as the clinical corpus, re-skinned:
  continuous signal      -> daily LMS engagement (minutes active)
  intermittent measures  -> quiz scores, taken on one day, graded on another
  delayed narrative      -> counselor notes, written well after a student
                             started struggling, not when it began
  uncertain onset         -> "started struggling" is a window, not an instant,
                             exactly like true_onset_time in the ICU corpus
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(11)

N_STUDENTS = 20
COURSE_START = pd.Timestamp("2026-02-02T00:00:00")
COURSE_WEEKS = 10
N_STRUGGLING = 8  # of 20, needs_intervention = 1

student_ids = [f"STU{i:03d}" for i in range(1, N_STUDENTS + 1)]
struggling = set(rng.choice(student_ids, size=N_STRUGGLING, replace=False))


def gen_engagement(sid, is_struggling):
    rows = []
    decline_start_day = None
    if is_struggling:
        decline_start_day = int(rng.uniform(15, 45))  # sometime in the middle of the course
    for day in range(COURSE_WEEKS * 7):
        ts = COURSE_START + pd.Timedelta(days=day, hours=int(rng.uniform(14, 21)))
        if is_struggling and decline_start_day is not None and day >= decline_start_day:
            days_since = day - decline_start_day
            baseline = max(5, 45 - days_since * 1.8)
        else:
            baseline = 45
        minutes = max(0, rng.normal(baseline, 9))
        # some days genuinely have no login at all, real missingness, not noise
        if rng.random() < (0.35 if is_struggling and decline_start_day and day >= decline_start_day else 0.08):
            continue
        rows.append(dict(student_id=sid, timestamp=ts, minutes_active=round(minutes, 1)))
    return rows, decline_start_day


def gen_assessments(sid, decline_start_day):
    rows = []
    for week in range(COURSE_WEEKS):
        day = week * 7 + int(rng.uniform(2, 5))
        assess_time = COURSE_START + pd.Timedelta(days=day, hours=int(rng.uniform(9, 15)))
        result_time = assess_time + pd.Timedelta(hours=float(rng.uniform(20, 96)))  # grading lag
        if decline_start_day is not None and day >= decline_start_day:
            score = max(20, rng.normal(58, 12))
        else:
            score = min(100, rng.normal(84, 8))
        rows.append(dict(student_id=sid, assess_time=assess_time, result_time=result_time,
                          score=round(score, 1)))
    return rows


NOTE_TEMPLATES = {
    "routine_checkin": "Routine check-in. Student engagement and performance within expected range.",
    "concern_flagged": "Noticed declining engagement and quiz performance over recent weeks. Flagging for follow-up.",
    "intervention": "Met with student to discuss course difficulties. Support plan put in place.",
}


def gen_notes(sid, decline_start_day):
    rows = []
    note_id = 1
    # a routine check-in early on, for everyone
    rows.append(dict(student_id=sid, note_id=f"{sid}-N{note_id}", note_type="routine_checkin",
                      event_time=COURSE_START + pd.Timedelta(days=10),
                      storetime=COURSE_START + pd.Timedelta(days=10, hours=float(rng.uniform(1, 6))),
                      text=NOTE_TEMPLATES["routine_checkin"]))
    note_id += 1
    if decline_start_day is not None:
        # the flag is written well AFTER the decline actually started, documentation
        # lag, exactly the ICU corpus's central problem, now in a classroom
        flag_lag_days = float(rng.uniform(4, 12))
        event_time = COURSE_START + pd.Timedelta(days=decline_start_day)
        storetime = event_time + pd.Timedelta(days=flag_lag_days)
        rows.append(dict(student_id=sid, note_id=f"{sid}-N{note_id}", note_type="concern_flagged",
                          event_time=event_time, storetime=storetime,
                          text=NOTE_TEMPLATES["concern_flagged"]))
        note_id += 1
        meeting_time = storetime + pd.Timedelta(days=float(rng.uniform(1, 3)))
        intervention_storetime = meeting_time + pd.Timedelta(hours=float(rng.uniform(1, 4)))
        rows.append(dict(student_id=sid, note_id=f"{sid}-N{note_id}", note_type="intervention",
                          event_time=meeting_time, storetime=intervention_storetime,
                          text=NOTE_TEMPLATES["intervention"]))
    return rows


def build(out_dir="."):
    engagement_rows, assess_rows, note_rows, label_rows = [], [], [], []
    for sid in student_ids:
        is_struggling = sid in struggling
        eng, decline_day = gen_engagement(sid, is_struggling)
        engagement_rows += eng
        assess_rows += gen_assessments(sid, decline_day)
        note_rows += gen_notes(sid, decline_day)

        if decline_day is not None:
            true_onset = COURSE_START + pd.Timedelta(days=decline_day)
            window_start = true_onset - pd.Timedelta(days=2)
            window_end = true_onset + pd.Timedelta(days=5)
        else:
            true_onset = window_start = window_end = pd.NaT
        label_rows.append(dict(student_id=sid, needs_intervention=int(is_struggling),
                                true_onset_time=true_onset,
                                possible_window_start=window_start, possible_window_end=window_end))

    engagement = pd.DataFrame(engagement_rows)
    assessments = pd.DataFrame(assess_rows)
    notes = pd.DataFrame(note_rows)
    labels = pd.DataFrame(label_rows)

    def round_minute(df, cols):
        for c in cols:
            df[c] = pd.to_datetime(df[c]).dt.round("min")
        return df

    engagement = round_minute(engagement, ["timestamp"])
    assessments = round_minute(assessments, ["assess_time", "result_time"])
    notes = round_minute(notes, ["event_time", "storetime"])
    labels = round_minute(labels, ["true_onset_time", "possible_window_start", "possible_window_end"])

    engagement.to_csv(f"{out_dir}/education_engagement.csv", index=False)
    assessments.to_csv(f"{out_dir}/education_assessments.csv", index=False)
    notes.to_csv(f"{out_dir}/education_notes.csv", index=False)
    labels.to_csv(f"{out_dir}/education_labels.csv", index=False)
    print(f"{len(student_ids)} students, {N_STRUGGLING} needing intervention, "
          f"{len(engagement)} engagement rows, {len(assessments)} assessments, {len(notes)} notes")
    return engagement, assessments, notes, labels


if __name__ == "__main__":
    build(".")
