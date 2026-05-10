# Machine learning model functions for anomaly detection

from typing import Dict, List, Optional, Union, Tuple, Set
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import FeatureHasher

from config import FEATURE_COLS, HASH_FEATURES, DEFAULT_CONTAMINATION, DEFAULT_N_ESTIMATORS, RANDOM_STATE, HIGH_THRESHOLD_DEFAULT, WARN_THRESHOLD_DEFAULT, TOP_ROOT_CAUSES_DEFAULT, ERROR_CODES, ROOT_CAUSE_RULES, BEHAVIOR_GROUPS
from data_models import PassModelBundle, ParsedLog


def train_pass_iforest_model(passed_logs: List[ParsedLog],
                             contamination: float = DEFAULT_CONTAMINATION,
                             n_estimators: int = DEFAULT_N_ESTIMATORS,
                             random_state: int = RANDOM_STATE) -> PassModelBundle:
    """
    Train an IsolationForest model on passed log data for anomaly detection.

    Args:
        passed_logs: List of successfully parsed logs
        contamination: Expected proportion of outliers in training data
        n_estimators: Number of trees in the forest
        random_state: Random seed for reproducibility

    Returns:
        Trained model bundle with scaler, model, hasher, and score thresholds
    """
    from data_processing import _build_training_matrix

    df_train, X_train, hasher, pass_signal_set = _build_training_matrix(passed_logs)

    if X_train.size == 0:
        # Fallback empty model bundle for edge cases
        scaler = StandardScaler()
        model = IsolationForest(random_state=random_state)
        return PassModelBundle(scaler, model, hasher, 0.0, 1.0, pass_signal_set)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state
    )
    model.fit(Xs)

    # Scale scores to 0-100 range using training distribution percentiles
    train_scores = -model.score_samples(Xs)
    p05 = float(np.percentile(train_scores, 5))
    p95 = float(np.percentile(train_scores, 95))
    if abs(p95 - p05) < 1e-9:
        p95 = p05 + 1e-6

    return PassModelBundle(
        scaler=scaler,
        model=model,
        hasher=hasher,
        score_p05=p05,
        score_p95=p95,
        pass_signal_set=pass_signal_set
    )


def _score_row_with_model(signal: str,
                          row_features: pd.Series,
                          bundle: PassModelBundle,
                          feature_cols: List[str] = FEATURE_COLS) -> float:
    """Score a signal's features using the trained anomaly detection model."""
    # Extract numeric features
    x_num = row_features[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy().reshape(1, -1)

    # Hash the signal name for categorical feature
    x_sig = bundle.hasher.transform([[str(signal)]]).toarray()

    # Combine features and scale
    x = np.hstack([x_num, x_sig])
    xs = bundle.scaler.transform(x)

    # Get anomaly score (higher = more anomalous)
    raw_score = -bundle.model.score_samples(xs)[0]

    # Scale to 0-100 range using training distribution
    scaled_score = 100.0 * (raw_score - bundle.score_p05) / (bundle.score_p95 - bundle.score_p05)
    return float(np.clip(scaled_score, 0.0, 100.0))


def build_reference_profile(passed_logs: List[ParsedLog]) -> pd.DataFrame:
    """Build a statistical reference profile from passed logs."""
    all_features = []
    for log in passed_logs:
        f = log.features.copy()
        f["log_name"] = log.name
        all_features.append(f)

    if not all_features:
        return pd.DataFrame()

    df = pd.concat(all_features, ignore_index=True)
    profile_rows = []

    for signal, grp in df.groupby("signal"):
        row = {"signal": signal}
        for col in FEATURE_COLS:
            if col not in grp.columns:
                continue
            vals = pd.to_numeric(grp[col], errors="coerce").dropna()
            if len(vals) == 0:
                row[f"{col}_ref_median"] = np.nan
                row[f"{col}_ref_mad"] = np.nan
                row[f"{col}_ref_min"] = np.nan
                row[f"{col}_ref_max"] = np.nan
            else:
                median = vals.median()
                mad = np.median(np.abs(vals - median))  # Median Absolute Deviation
                row[f"{col}_ref_median"] = median
                row[f"{col}_ref_mad"] = mad
                row[f"{col}_ref_min"] = vals.min()
                row[f"{col}_ref_max"] = vals.max()
        profile_rows.append(row)

    return pd.DataFrame(profile_rows)


def robust_z(value: float, median: float, mad: float) -> float:
    """Calculate robust Z-score using median and MAD (Median Absolute Deviation)."""
    if pd.isna(value) or pd.isna(median):
        return np.nan
    scale = 1.4826 * mad  # Scale factor for MAD to be consistent with std
    if pd.isna(scale) or scale < 1e-9:
        return 0.0 if abs(value - median) < 1e-9 else abs(value - median)
    return abs(value - median) / scale


def calculate_divergence_score(feature_vals: pd.Series, ref_median: float, ref_mad: float) -> Tuple[float, List[str]]:
    """Calculate divergence score and collect reasons for a feature set."""
    scores = []
    reasons = []

    for col in FEATURE_COLS:
        val = pd.to_numeric(pd.Series([feature_vals.get(col)]), errors="coerce").iloc[0]
        med = ref_median.get(f"{col}_ref_median", np.nan) if isinstance(ref_median, pd.Series) else np.nan
        mad = ref_mad.get(f"{col}_ref_mad", np.nan) if isinstance(ref_mad, pd.Series) else np.nan

        z = robust_z(val, med, mad)
        if not pd.isna(z):
            scores.append(min(z, 25.0))

            if z >= 6:
                reasons.append(f"{col} deviates strongly")
            elif z >= 3:
                reasons.append(f"{col} deviates")

    score = float(np.nanmax(scores)) if scores else 0.0
    return score, reasons


def determine_status_and_severity(score: float) -> Tuple[str, str]:
    """Determine status emoji and severity level based on divergence score."""
    if score >= HIGH_THRESHOLD_DEFAULT:
        return "❌", "High"
    elif score >= WARN_THRESHOLD_DEFAULT:
        return "⚠️", "Medium"
    else:
        return "✅", "Low"


def rule_based_boost(signal: str, row: pd.Series) -> Tuple[float, List[str]]:
    """Apply rule-based boosts for known problematic signal patterns."""
    boost = 0.0
    reasons = []

    # Extract numeric features safely
    features = {}
    for feat in ["last", "mean", "span", "zero_frac", "one_frac", "transitions"]:
        features[feat] = pd.to_numeric(pd.Series([row.get(feat)]), errors="coerce").iloc[0]

    s = signal.lower()

    # Rule 1: Inverted/auth invalid flags persistently high
    if "_inv" in s and not pd.isna(features["one_frac"]) and features["one_frac"] > 0.8:
        boost += 45
        reasons.append("inverted/auth invalid flag is persistently high")

    # Rule 2: Wheel angular velocity auth remains invalid/zero
    if "angvelauth" in s and not pd.isna(features["zero_frac"]) and features["zero_frac"] > 0.8:
        boost += 25
        reasons.append("wheel angular velocity auth remains invalid/zero")

    # Rule 3: Speed signals remain zero/static
    if any(speed in s for speed in ["flspeed", "frspeed", "speed"]) and not pd.isna(features["zero_frac"]) and features["zero_frac"] > 0.95:
        boost += 15
        reasons.append("speed remains zero/static")

    # Rule 4: HIL motion input remains static zero
    if "gkn_hil.panel" in s and not pd.isna(features["zero_frac"]) and features["zero_frac"] > 0.95:
        boost += 10
        reasons.append("HIL motion input remains static zero")

    # Rule 5: RBS message transmission is unstable/toggling
    if "ign_txing_rbs_msgs" in s and not pd.isna(features["transitions"]) and features["transitions"] > 10:
        boost += 20
        reasons.append("RBS message transmission is unstable/toggling")

    # Rule 6: CAN status reports error
    if "can" in s and "status" in s:
        raw_value = str(row.get("last", "")).lower()
        if "error" in raw_value:
            boost += 35
            reasons.append("CAN status reports error active")

    # Rule 7: Signal is stuck at zero
    if not pd.isna(features["span"]) and features["span"] == 0 and not pd.isna(features["zero_frac"]) and features["zero_frac"] > 0.95:
        boost += 5
        reasons.append("signal is stuck at zero")

    return boost, reasons


def compare_to_reference(log: ParsedLog, ref: pd.DataFrame) -> pd.DataFrame:
    """Compare log features to reference profile and identify divergences."""
    if log.features.empty or ref.empty:
        return pd.DataFrame()

    merged = log.features.merge(ref, on="signal", how="outer", indicator=True)
    rows = []

    for _, r in merged.iterrows():
        signal = r["signal"]
        missing_in_failed = r["_merge"] == "right_only"
        new_in_failed = r["_merge"] == "left_only"

        if missing_in_failed:
            rows.append({
                "log_name": log.name,
                "signal": signal,
                "divergence_score": 100.0,
                "status": "❌",
                "reason": "Signal exists in passed reference but is missing in failed log",
                "severity": "High"
            })
            continue

        if new_in_failed:
            rows.append({
                "log_name": log.name,
                "signal": signal,
                "divergence_score": 75.0,
                "status": "⚠️",
                "reason": "Signal appears in failed log but not in passed reference",
                "severity": "Medium"
            })
            continue

        # Calculate divergence for common signals
        score, reasons = calculate_divergence_score(r, r, r)  # ref data is in the same row

        # Apply rule-based boosts
        rule_boost, rule_reasons = rule_based_boost(signal, r)
        score = min(100.0, score * 4.0 + rule_boost)
        reasons.extend(rule_reasons)

        reason = "; ".join(sorted(set(reasons))) if reasons else "Matches passed reference behavior"
        status, severity = determine_status_and_severity(score)

        rows.append({
            "log_name": log.name,
            "signal": signal,
            "divergence_score": score,
            "status": status,
            "reason": reason,
            "severity": severity
        })

    return pd.DataFrame(rows)


def compare_to_pass_model(log: ParsedLog,
                          ref: pd.DataFrame,
                          bundle: PassModelBundle,
                          feature_cols: List[str] = FEATURE_COLS) -> pd.DataFrame:
    """
    Compare log to ML model trained on passed logs.

    Produces the same output schema as compare_to_reference(),
    but uses ML-based anomaly scoring instead of statistical comparison.
    """
    if log.features.empty:
        return pd.DataFrame()

    ref_signals = set(ref["signal"].astype(str).unique()) if not ref.empty else set()
    fail_signals = set(log.features["signal"].astype(str).unique())
    rows = []

    # Handle missing signals (present in PASS ref but not in FAIL)
    for signal in sorted(ref_signals - fail_signals):
        rows.append({
            "log_name": log.name,
            "signal": signal,
            "divergence_score": 100.0,
            "status": "❌",
            "reason": "Signal exists in passed reference but is missing in failed log",
            "severity": "High"
        })

    # Handle new signals (present in FAIL but not in PASS ref)
    for signal in sorted(fail_signals - ref_signals):
        rows.append({
            "log_name": log.name,
            "signal": signal,
            "divergence_score": 75.0,
            "status": "⚠️",
            "reason": "Signal appears in failed log but not in passed reference",
            "severity": "Medium"
        })

    # Score common signals using ML model
    common_signals = log.features[log.features["signal"].isin(ref_signals)].copy() if ref_signals else log.features.copy()
    merged = common_signals.merge(ref, on="signal", how="left") if not ref.empty else common_signals

    for _, row in merged.iterrows():
        signal = row["signal"]

        # Get ML anomaly score
        score = _score_row_with_model(signal, row, bundle, feature_cols=feature_cols)

        # Generate human-readable reasons based on reference profile
        _, reasons = calculate_divergence_score(row, row, row)
        reason = "; ".join(sorted(set(reasons))) if reasons else "Matches passed reference behavior"

        status, severity = determine_status_and_severity(score)

        rows.append({
            "log_name": log.name,
            "signal": signal,
            "divergence_score": float(score),
            "status": status,
            "reason": reason,
            "severity": severity
        })

    return pd.DataFrame(rows)


def identify_root_causes(divergence_df: pd.DataFrame, top_n: int = TOP_ROOT_CAUSES_DEFAULT) -> pd.DataFrame:
    """Identify probable root causes by mapping high-divergence signals to error codes."""
    if divergence_df.empty:
        return pd.DataFrame()

    # Focus on top 50 highest-scoring signals for efficiency
    high_divergence_signals = divergence_df.sort_values("divergence_score", ascending=False).head(50)
    root_cause_rows = []

    for _, row in high_divergence_signals.iterrows():
        signal = str(row["signal"]).lower()
        matched = False

        # Check against predefined root cause rules
        for rule in ROOT_CAUSE_RULES:
            if all(pattern in signal for pattern in rule["pattern"]):
                for error_bit in rule["error_bits"]:
                    error_info = ERROR_CODES.get(error_bit, {})
                    root_cause_rows.append({
                        "log_name": row["log_name"],
                        "signal": row["signal"],
                        "divergence_score": row["divergence_score"],
                        "status": row["status"],
                        "probable_root_cause": rule["root_cause"],
                        "interpretation": rule["interpretation"],
                        "mapped_error_bit": error_bit,
                        "mapped_error_name": error_info.get("name", ""),
                        "mapped_error_status_value": error_info.get("status_value", ""),
                        "error_description": error_info.get("description", ""),
                        "evidence": row["reason"],
                    })
                matched = True

        # Handle unmapped high-divergence signals
        if not matched and row["divergence_score"] >= HIGH_THRESHOLD_DEFAULT:
            root_cause_rows.append({
                "log_name": row["log_name"],
                "signal": row["signal"],
                "divergence_score": row["divergence_score"],
                "status": row["status"],
                "probable_root_cause": "Unmapped high-divergence signal",
                "interpretation": "Signal diverges strongly from passed reference but no specific rule matched.",
                "mapped_error_bit": None,
                "mapped_error_name": "Review manually",
                "mapped_error_status_value": "",
                "error_description": "",
                "evidence": row["reason"],
            })

    root_df = pd.DataFrame(root_cause_rows)
    if root_df.empty:
        return root_df

    # Rank by divergence score
    root_df["rank_score"] = root_df["divergence_score"]
    root_df = root_df.sort_values("rank_score", ascending=False)
    return root_df.head(top_n)


def behavior_for_signal(signal: str) -> str:
    """Categorize signal into behavioral groups for matrix display."""
    s = signal.lower()

    # Check predefined behavior groups
    for behavior, keywords in BEHAVIOR_GROUPS.items():
        if all(keyword in s for keyword in keywords):
            return behavior

    # Fallback categorization
    if "wheel" in s:
        return "Wheel speeds / motion inputs"
    if "can" in s:
        return "CAN / CAN-FD health"
    if "ign" in s:
        return "Ignition / RBS"
    if "gkn_hil" in s:
        return "HIL motion panel"

    return "Other signals"


def build_divergence_matrix(divergence_df: pd.DataFrame) -> pd.DataFrame:
    """Build a matrix view of divergence by behavior categories and logs."""
    if divergence_df.empty:
        return pd.DataFrame()

    df = divergence_df.copy()
    df["behavior"] = df["signal"].apply(behavior_for_signal)

    # Aggregate by behavior and log
    agg = (
        df.groupby(["behavior", "log_name"])
        .agg(
            max_score=("divergence_score", "max"),
            top_signal=("signal", lambda x: x.iloc[0] if len(x) > 0 else ""),
            reasons=("reason", lambda x: "; ".join(pd.Series(x).dropna().astype(str).head(2)))
        )
        .reset_index()
    )

    # Add status and cell formatting
    agg["status"] = agg["max_score"].apply(lambda score: determine_status_and_severity(score)[0])
    agg["cell"] = agg.apply(lambda r: f"{r['status']} ({r['max_score']:.0f})", axis=1)

    # Pivot to matrix format
    matrix = agg.pivot(index="behavior", columns="log_name", values="cell").fillna("✅")
    matrix = matrix.reset_index().rename(columns={"behavior": "Signal / Behavior"})

    # Add interpretation column
    interpretations = []
    for behavior in matrix["Signal / Behavior"]:
        behavior_data = agg[agg["behavior"] == behavior].sort_values("max_score", ascending=False)
        if behavior_data.empty:
            interpretations.append("Matches passed reference behavior")
        else:
            top_score = behavior_data.iloc[0]["max_score"]
            top_reason = behavior_data.iloc[0]["reasons"]
            if top_score >= HIGH_THRESHOLD_DEFAULT:
                interpretations.append(f"Divergent from passed reference: {top_reason}")
            elif top_score >= WARN_THRESHOLD_DEFAULT:
                interpretations.append(f"Transitional / unstable: {top_reason}")
            else:
                interpretations.append("Matches passed reference behavior")

    matrix["Interpretation"] = interpretations
    return matrix


def heatmap_from_divergence(divergence_df: pd.DataFrame):
    """Create a heatmap visualization of signal divergence across logs."""
    if divergence_df.empty:
        return None

    # Select top 40 most divergent signals for visualization
    top_signals = (
        divergence_df.groupby("signal")["divergence_score"]
        .max()
        .sort_values(ascending=False)
        .head(40)
        .index
    )

    # Filter to top signals and prepare for heatmap
    heatmap_data = divergence_df[divergence_df["signal"].isin(top_signals)].copy()
    heatmap_data["signal_short"] = heatmap_data["signal"].str.slice(0, 30)

    # Create pivot table for heatmap
    pivot = heatmap_data.pivot_table(
        values="divergence_score",
        index="signal_short",
        columns="log_name",
        aggfunc="max",
        fill_value=0
    )

    # Create heatmap
    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale="RdYlGn_r",
        title="Signal Divergence Heatmap (Top 40 Most Divergent Signals)",
        labels={"x": "Log File", "y": "Signal", "color": "Divergence Score"}
    )

    fig.update_layout(
        xaxis_tickangle=-45,
        height=max(400, len(pivot) * 20)
    )

    return fig


def isolation_forest_log_anomaly(all_logs: List[ParsedLog]) -> pd.DataFrame:
    """Apply IsolationForest to detect anomalous logs at the log level."""
    if len(all_logs) < 3:
        return pd.DataFrame()

    # Aggregate features across all signals for each log
    log_features = []
    for log in all_logs:
        if log.features.empty:
            continue

        # Aggregate signal-level features to log-level
        agg_features = {}
        for col in ["count", "numeric_count", "mean", "std", "min", "max", "span", "zero_frac", "one_frac", "transitions"]:
            if col in log.features.columns:
                vals = pd.to_numeric(log.features[col], errors="coerce").dropna()
                if len(vals) > 0:
                    if col in ["count", "numeric_count", "transitions"]:
                        agg_features[f"log_{col}"] = vals.sum()
                    else:
                        agg_features[f"log_{col}_mean"] = vals.mean()
                        agg_features[f"log_{col}_std"] = vals.std(ddof=0) if len(vals) > 1 else 0

        agg_features["log_name"] = log.name
        agg_features["label"] = log.label
        agg_features["total_signals"] = len(log.features)
        agg_features["total_events"] = len(log.events)

        log_features.append(agg_features)

    if len(log_features) < 3:
        return pd.DataFrame()

    df = pd.DataFrame(log_features)

    # Prepare features for ML
    feature_cols = [c for c in df.columns if c not in ["log_name", "label"]]
    X = df[feature_cols].fillna(0).to_numpy()

    # Train IsolationForest
    model = IsolationForest(n_estimators=100, contamination=0.1, random_state=RANDOM_STATE)
    scores = model.fit_predict(X)
    anomaly_scores = -model.score_samples(X)

    # Scale scores to 0-100
    score_min, score_max = anomaly_scores.min(), anomaly_scores.max()
    if score_max > score_min:
        scaled_scores = 100 * (anomaly_scores - score_min) / (score_max - score_min)
    else:
        scaled_scores = np.zeros(len(anomaly_scores))

    df["iforest_score"] = scaled_scores
    df["iforest_flag"] = scores  # -1 for anomaly, 1 for normal

    return df[["log_name", "label", "iforest_score", "iforest_flag", "total_signals", "total_events"]]