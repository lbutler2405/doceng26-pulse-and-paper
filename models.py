"""
models.py

Block 2: the naive fixed-window baseline vs. the time-aware model, built on
the exact same underlying data, differing only in how they turn raw
timestamps into features. Both predict the same thing: will this patient
deteriorate at some point during their stay (`outcome`), using only data
available by a fixed prediction time.

Neither model can see the future. The naive model's problem is what it
leans on: a blunt "how many notes / how recently" signal that quietly
encodes documentation speed rather than physiology, one that happens to
correlate with the outcome in-sample but degrades unevenly across patients
whose documentation is slower for structural reasons (Part 4: population
and socioeconomic context).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, recall_score

import alignment as al

PREDICTION_HOUR = 48  # fixed window, same for every patient, the naive design choice itself


def _load(data_dir="."):
    patients = pd.read_csv(f"{data_dir}/synthetic_patients.csv", parse_dates=["admittime", "dischtime"])
    vitals = pd.read_csv(f"{data_dir}/synthetic_vitals_continuous.csv", parse_dates=["charttime"])
    labs = pd.read_csv(f"{data_dir}/synthetic_labs_intermittent.csv", parse_dates=["charttime", "storetime"])
    notes = pd.read_csv(f"{data_dir}/synthetic_notes.csv", parse_dates=["event_charttime", "storetime"])
    labels = pd.read_csv(f"{data_dir}/synthetic_labels.csv", parse_dates=["true_onset_time"])
    return patients, vitals, labs, notes, labels


PREDICTION_OFFSET_HOURS = 6.0  # evaluate this many hours after the (unknown-to-the-model) onset point.
                                 # Late enough that documentation has had a chance to catch up for
                                 # *fast*-charting patients, early enough that it hasn't for slow ones,
                                 # which is exactly the gap this block is built to expose.


def _prediction_times(patients, labels, seed=42):
    """For outcome=1 patients: true_onset + offset. For outcome=0 patients: an
    equally-plausible snapshot time, drawn from the same relative-onset-timing
    distribution observed in the positives, so prediction time isn't itself a
    giveaway of the label."""
    rng = np.random.default_rng(seed)
    l = labels.set_index("subject_id")
    p = patients.set_index("subject_id")

    pos_fracs = []
    for sid in l[l.outcome == 1].index:
        los_h = (p.loc[sid].dischtime - p.loc[sid].admittime).total_seconds() / 3600
        onset_h = (l.loc[sid].true_onset_time - p.loc[sid].admittime).total_seconds() / 3600
        pos_fracs.append(onset_h / los_h)
    pos_fracs = np.array(pos_fracs)

    times = {}
    for sid in p.index:
        los_h = (p.loc[sid].dischtime - p.loc[sid].admittime).total_seconds() / 3600
        if l.loc[sid].outcome == 1:
            t = l.loc[sid].true_onset_time + pd.Timedelta(hours=PREDICTION_OFFSET_HOURS)
        else:
            frac = rng.choice(pos_fracs)
            t = p.loc[sid].admittime + pd.Timedelta(hours=los_h * frac + PREDICTION_OFFSET_HOURS)
        times[sid] = min(t, p.loc[sid].dischtime)  # never past discharge
    return times


def build_shortcut_features(data_dir=".", lookback_hours=48):
    """Documentation-timing signal ONLY, no physiology at all. Used as a
    standalone probe: how much can charting timing alone tell you, and who
    does it fail?"""
    patients, vitals, labs, notes, labels = _load(data_dir)
    pred_times = _prediction_times(patients, labels)
    rows = []
    for p in patients.itertuples():
        sid = p.subject_id
        pn = notes[notes.subject_id == sid]
        sig = al.naive_note_signal(pn, pred_times[sid], lookback_hours)
        rows.append(dict(subject_id=sid, **sig))
    return pd.DataFrame(rows).set_index("subject_id")


def build_features(data_dir=".", lookback_hours=48):
    """Returns (naive_df, time_aware_df, meta_df), all indexed the same way,
    meta_df carries subject_id, outcome, and the tier/context columns used
    for subgroup evaluation."""
    patients, vitals, labs, notes, labels = _load(data_dir)
    labels_by_id = labels.set_index("subject_id")
    pred_times = _prediction_times(patients, labels)

    naive_rows, aware_rows, meta_rows = [], [], []

    for p in patients.itertuples():
        sid = p.subject_id
        pred_time = pred_times[sid]

        pv = vitals[vitals.subject_id == sid]
        pl = labs[labs.subject_id == sid]
        pn = notes[notes.subject_id == sid]

        # --- naive: last snapshot values only, plus the note-count shortcut ---
        hr_last, _, _ = al.event_centered_trend(pv, "charttime", "heartrate", pred_time, lookback_hours)
        o2_last, _, _ = al.event_centered_trend(pv, "charttime", "o2sat", pred_time, lookback_hours)
        sbp_last, _, _ = al.event_centered_trend(pv, "charttime", "sbp", pred_time, lookback_hours)
        lac = pl[pl.label == "Lactate"]
        trop = pl[pl.label == "Troponin"]
        lac_last, _, _ = al.event_centered_trend(lac, "storetime", "value", pred_time, lookback_hours)
        trop_last, _, _ = al.event_centered_trend(trop, "storetime", "value", pred_time, lookback_hours)
        n_labs = len(al.naive_window(pl, "storetime", pred_time, lookback_hours))
        note_signal = al.naive_note_signal(pn, pred_time, lookback_hours)

        naive_rows.append({
            "subject_id": sid,
            "heartrate_last": hr_last, "o2sat_last": o2_last, "sbp_last": sbp_last,
            "lactate_last": lac_last, "troponin_last": trop_last, "n_labs": n_labs,
            **note_signal,
        })

        # --- time-aware: trend features + missingness, no note shortcut ---
        hr_last2, hr_slope, hr_n = al.event_centered_trend(pv, "charttime", "heartrate", pred_time, lookback_hours)
        o2_last2, o2_slope, o2_n = al.event_centered_trend(pv, "charttime", "o2sat", pred_time, lookback_hours)
        sbp_last2, sbp_slope, sbp_n = al.event_centered_trend(pv, "charttime", "sbp", pred_time, lookback_hours)
        lac_last2, lac_slope, lac_n = al.event_centered_trend(lac, "storetime", "value", pred_time, lookback_hours)
        trop_last2, trop_slope, trop_n = al.event_centered_trend(trop, "storetime", "value", pred_time, lookback_hours)

        n_vitals_obs, vitals_ratio = al.missingness_features(pv, "charttime", pred_time, lookback_hours,
                                                               expected_per_day=24 * 60 / 30)
        n_labs_obs, labs_ratio = al.missingness_features(pl, "storetime", pred_time, lookback_hours,
                                                           expected_per_day=3)

        aware_rows.append({
            "subject_id": sid,
            "heartrate_last": hr_last2, "heartrate_slope": hr_slope,
            "o2sat_last": o2_last2, "o2sat_slope": o2_slope,
            "sbp_last": sbp_last2, "sbp_slope": sbp_slope,
            "lactate_last": lac_last2, "lactate_slope": lac_slope,
            "troponin_last": trop_last2, "troponin_slope": trop_slope,
            "vitals_completeness": vitals_ratio, "labs_completeness": labs_ratio,
        })

        lrow = labels_by_id.loc[sid]
        meta_rows.append({
            "subject_id": sid, "outcome": lrow.outcome,
            "hospital_complexity_tier": lrow.hospital_complexity_tier,
            "entanglement_tier": lrow.entanglement_tier,
            "population_context": lrow.population_context,
            "socioeconomic_context": lrow.socioeconomic_context,
        })

    naive_df = pd.DataFrame(naive_rows).set_index("subject_id")
    aware_df = pd.DataFrame(aware_rows).set_index("subject_id")
    meta_df = pd.DataFrame(meta_rows).set_index("subject_id")
    return naive_df, aware_df, meta_df


def make_pipeline():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])


def train_and_predict(X, y, seed=42):
    """Simple train/test split, returns (pipeline, X_test, y_test, y_pred_proba)."""
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    pipe = make_pipeline()
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    return pipe, X_test, y_test, proba


def feature_importance(pipe, feature_names):
    coefs = pipe.named_steps["clf"].coef_[0]
    return pd.Series(coefs, index=feature_names).sort_values(key=abs, ascending=False)


def subgroup_recall(pipe, X, y, meta, subgroup_col, threshold=0.5):
    proba = pipe.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    df = pd.DataFrame({"y": y.values, "pred": pred, "group": meta.loc[X.index, subgroup_col].values})
    rows = []
    for g, gdf in df.groupby("group"):
        pos = gdf[gdf.y == 1]
        if len(pos) == 0:
            continue
        rows.append(dict(group=g, n_positive=len(pos), recall=recall_score(pos.y, pos.pred), n_total=len(gdf)))
    return pd.DataFrame(rows).sort_values("group")


if __name__ == "__main__":
    naive_df, aware_df, meta_df = build_features(".")
    shortcut_df = build_shortcut_features(".")
    y = meta_df["outcome"]

    print("=== The shortcut, isolated: documentation timing ONLY, no physiology ===")
    pipe_s, Xte_s, yte_s, proba_s = train_and_predict(shortcut_df, y)
    print("AUROC:", round(roc_auc_score(yte_s, proba_s), 3))
    print("\nSubgroup recall (entanglement_tier):")
    print(subgroup_recall(pipe_s, shortcut_df, y, meta_df, "entanglement_tier"))
    print("\nSubgroup recall (population_context):")
    print(subgroup_recall(pipe_s, shortcut_df, y, meta_df, "population_context"))

    print("\n=== Naive model: physiology + the shortcut, combined ===")
    pipe_n, Xte_n, yte_n, proba_n = train_and_predict(naive_df, y)
    print("AUROC:", round(roc_auc_score(yte_n, proba_n), 3))
    print(feature_importance(pipe_n, naive_df.columns))

    print("\n=== Time-aware model: trend + missingness, the shortcut never offered ===")
    pipe_a, Xte_a, yte_a, proba_a = train_and_predict(aware_df, y)
    print("AUROC:", round(roc_auc_score(yte_a, proba_a), 3))
    print(feature_importance(pipe_a, aware_df.columns))
