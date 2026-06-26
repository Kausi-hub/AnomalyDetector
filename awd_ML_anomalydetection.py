import re
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ======================
# CONFIG
# ======================
st.set_page_config(page_title="EOL Anomaly Dashboard", layout="wide")

DEFAULT_CONTAMINATION = 0.05
RANDOM_STATE = 42

# ======================
# UTILITIES
# ======================

def read_uploaded_file(file):
    return file.read().decode("utf-8", errors="ignore")


def normalize_signal_name(name):
    name = name.strip()
    name = name.replace("::", ".")
    return re.sub(r"\s+", "_", name)


def parse_log(text, name, label):
    rows = []
    for line in text.splitlines():
        m = re.match(r"^\s*(\d+\.\d+)\s+(.*)$", line)
        if not m:
            continue

        time = float(m.group(1))
        payload = m.group(2)

        match = re.match(r"(.+?):=(.+)", payload)
        if match:
            rows.append({
                "time": time,
                "signal": normalize_signal_name(match.group(1)),
                "value": match.group(2).strip(),
                "log": name,
                "label": label
            })

    df = pd.DataFrame(rows)

    if df.empty:
        return df, pd.DataFrame()

    features = df.groupby("signal").agg(
        count=("value", "count"),
        unique=("value", "nunique")
    ).reset_index()

    return df, features


# ======================
# MODEL
# ======================

@st.cache_resource
def train_model(features_df):
    if features_df.empty:
        return None, None

    X = features_df[["count", "unique"]]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = IsolationForest(
        contamination=DEFAULT_CONTAMINATION,
        random_state=RANDOM_STATE
    )
    model.fit(Xs)

    return model, scaler


def score_features(features_df, model, scaler):
    if features_df.empty or model is None:
        return features_df

    X = features_df[["count", "unique"]]
    Xs = scaler.transform(X)

    scores = -model.score_samples(Xs)
    features_df["score"] = scores
    return features_df


# ======================
# MAIN APP
# ======================

def main():
    st.title("🚗 EOL Validation Anomaly & Root Cause Dashboard")

    st.sidebar.header("Upload Logs")

    passed_files = st.sidebar.file_uploader(
        "Passed Logs",
        accept_multiple_files=True
    )

    failed_files = st.sidebar.file_uploader(
        "Failed Logs",
        accept_multiple_files=True
    )

    if not passed_files or not failed_files:
        st.info("Upload both passed and failed logs")
        return

    # ======================
    # PARSE LOGS
    # ======================
    passed_features = []
    failed_features = []

    for f in passed_files:
        text = read_uploaded_file(f)
        _, feat = parse_log(text, f.name, "passed")
        if not feat.empty:
            passed_features.append(feat)

    for f in failed_files:
        text = read_uploaded_file(f)
        _, feat = parse_log(text, f.name, "failed")
        if not feat.empty:
            failed_features.append(feat)

    if not passed_features:
        st.error("No valid passed logs parsed")
        return

    passed_df = pd.concat(passed_features, ignore_index=True)

    # ======================
    # TRAIN MODEL
    # ======================
    model, scaler = train_model(passed_df)

    # ======================
    # SCORE FAILED
    # ======================
    results = []
    for f_df in failed_features:
        scored = score_features(f_df, model, scaler)
        results.append(scored)

    if not results:
        st.error("No failed features parsed")
        return

    result_df = pd.concat(results, ignore_index=True)

    # ======================
    # UI
    # ======================
    st.subheader("📊 Signal Anomaly Scores")

    st.dataframe(result_df.sort_values("score", ascending=False))

    st.subheader("🔥 Top Anomalies")

    st.dataframe(result_df.nlargest(10, "score"))

    st.bar_chart(result_df.set_index("signal")["score"])


# ======================
# RUN
# ======================
if __name__ == "__main__":
    main()
