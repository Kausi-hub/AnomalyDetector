# Streamlit UI components for the Anomaly Detection Dashboard

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

from config import HIGH_THRESHOLD_DEFAULT, WARN_THRESHOLD_DEFAULT, TOP_ROOT_CAUSES_DEFAULT, ERROR_CODES
from data_processing import read_uploaded_file, parse_log_text
from ml_models import (
    build_reference_profile, train_pass_iforest_model, compare_to_pass_model,
    identify_root_causes, build_divergence_matrix, heatmap_from_divergence,
    isolation_forest_log_anomaly
)
from data_models import ParsedLog


def setup_page_config():
    """Configure the Streamlit page."""
    st.set_page_config(
        page_title="EOL Divergence Matrix & Root Cause Dashboard",
        layout="wide"
    )


def display_header():
    """Display the main header and description."""
    st.title("Anomaly Detection & Root Cause Dashboard")

    st.markdown(
        """
Upload **passed reference logs** and **failed logs**.  
The dashboard parses CANoe-style text logs, builds a passed reference profile,
detects failed-log divergence, maps likely error codes, and generates a
simultaneous divergence matrix.
"""
    )


def create_sidebar(passed_files, failed_files, high_threshold, warn_threshold, top_root_causes, signal_filter):
    """Create the sidebar with file uploaders and settings."""
    with st.sidebar:
        st.header("1. Select input logs")

        passed_files = st.file_uploader(
            "Passed reference logs",
            type=["txt", "log", "asc", "csv"],
            accept_multiple_files=True,
            help="Upload one or more passed EOL data logs."
        )

        failed_files = st.file_uploader(
            "Failed / suspect logs",
            type=["txt", "log", "asc", "csv"],
            accept_multiple_files=True,
            help="Upload one or more failed or suspect EOL data logs."
        )

        st.header("2. Detection settings")
        high_threshold = st.slider("High divergence threshold", 40, 90, HIGH_THRESHOLD_DEFAULT)
        warn_threshold = st.slider("Warning divergence threshold", 10, 50, WARN_THRESHOLD_DEFAULT)
        top_root_causes = st.slider("Root-cause rows to show", 3, 20, TOP_ROOT_CAUSES_DEFAULT)

        st.header("3. Signal filter")
        signal_filter = st.text_input(
            "Only include signals containing",
            value="",
            help="Example: WheelSpeeds, GKN_HIL, IGN, CAN, torque"
        )

    return passed_files, failed_files, high_threshold, warn_threshold, top_root_causes, signal_filter


@st.cache_data
def parse_logs_cached(files, label) -> list[ParsedLog]:
    """Parse multiple log files with caching."""
    logs = []
    for f in files:
        text = read_uploaded_file(f)
        logs.append(parse_log_text(text, f.name, label))
    return logs


def display_log_metadata(all_logs: list[ParsedLog]):
    """Display metadata for uploaded logs."""
    st.subheader("Uploaded logs")
    metadata_rows = []
    for log in all_logs:
        metadata_rows.append({
            "log_name": log.name,
            "label": log.label,
            "events_parsed": len(log.events),
            "signals_parsed": 0 if log.features.empty else log.features["signal"].nunique(),
            "date": log.metadata.get("date", ""),
            "version": log.metadata.get("version", ""),
            "measurement_uuid": log.metadata.get("measurement_uuid", "")
        })
    st.dataframe(pd.DataFrame(metadata_rows), use_container_width=True)


def display_kpi_metrics(failed_logs: list[ParsedLog], divergence_df: pd.DataFrame):
    """Display key performance indicator metrics."""
    high_count = int((divergence_df["severity"] == "High").sum())
    warn_count = int((divergence_df["severity"] == "Medium").sum())
    ok_count = int((divergence_df["severity"] == "Low").sum())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Failed logs", len(failed_logs))
    k2.metric("High divergences", high_count)
    k3.metric("Warnings", warn_count)
    k4.metric("Matched / low", ok_count)


def create_main_tabs():
    """Create the main content tabs."""
    tab_matrix, tab_root, tab_heatmap, tab_details, tab_model, tab_error = st.tabs([
        "Divergence Matrix",
        "Root Cause",
        "Heatmap",
        "Signal Details",
        "Log-Level Model",
        "Error Code Reference"
    ])
    return tab_matrix, tab_root, tab_heatmap, tab_details, tab_model, tab_error


def display_divergence_matrix_tab(divergence_df: pd.DataFrame):
    """Display the divergence matrix tab."""
    st.subheader("Simultaneous Divergence Matrix")
    st.markdown("Legend: ✅ matches passed reference behavior &nbsp;&nbsp; ❌ divergent from passed reference &nbsp;&nbsp; ⚠️ transitional / unstable.")

    matrix = build_divergence_matrix(divergence_df)
    st.dataframe(matrix, use_container_width=True, hide_index=True)

    csv_data = matrix.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download divergence matrix CSV",
        data=csv_data,
        file_name="divergence_matrix.csv",
        mime="text/csv"
    )


def display_root_cause_tab(divergence_df: pd.DataFrame, top_root_causes: int):
    """Display the root cause analysis tab."""
    st.subheader("Probable Root Cause Identification")

    root_df = identify_root_causes(divergence_df, top_n=top_root_causes)

    if root_df.empty:
        st.success("No high-confidence root cause identified from the current thresholds.")
    else:
        st.dataframe(root_df.drop(columns=["rank_score"], errors="ignore"), use_container_width=True)

        top_cause = root_df.iloc[0]
        st.markdown("### Top hypothesis")
        st.error(
            f"""
**Root cause:** {top_cause['probable_root_cause']}  
**Evidence signal:** `{top_cause['signal']}`  
**Mapped error:** Bit {top_cause['mapped_error_bit']} — {top_cause['mapped_error_name']}  
**Evidence:** {top_cause['evidence']}
"""
        )

        csv_data = root_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download root-cause report CSV",
            data=csv_data,
            file_name="root_cause_report.csv",
            mime="text/csv"
        )


def display_heatmap_tab(divergence_df: pd.DataFrame):
    """Display the heatmap visualization tab."""
    st.subheader("Signal Divergence Heatmap")

    fig = heatmap_from_divergence(divergence_df)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No heatmap available.")


def display_signal_details_tab(divergence_df: pd.DataFrame):
    """Display the signal-level divergence details tab."""
    st.subheader("Signal-Level Divergence Details")

    sort_options = ["divergence_score", "signal", "log_name", "severity"]
    sort_col = st.selectbox("Sort by", sort_options, index=0)

    st.dataframe(
        divergence_df.sort_values(sort_col, ascending=False),
        use_container_width=True,
        hide_index=True
    )

    csv_data = divergence_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download signal divergence detail CSV",
        data=csv_data,
        file_name="signal_divergence_detail.csv",
        mime="text/csv"
    )


def display_log_model_tab(all_logs: list[ParsedLog]):
    """Display the log-level anomaly model tab."""
    st.subheader("Log-Level Anomaly Model")

    model_df = isolation_forest_log_anomaly(all_logs)
    st.dataframe(model_df, use_container_width=True, hide_index=True)

    if len(model_df) >= 3 and "iforest_score" in model_df.columns:
        fig = px.bar(
            model_df,
            x="log_name",
            y="iforest_score",
            color="iforest_flag",
            title="Isolation Forest Log-Level Anomaly Score"
        )
        fig.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)


def display_error_reference_tab():
    """Display the error code reference tab."""
    st.subheader("Error Code Reference")

    error_code_data = [
        {
            "Error Bit": bit,
            "Error Status Value": info["status_value"],
            "Error Name": info["name"],
            "Description": info["description"],
            "Root Cause Hint": info["root_hint"],
        }
        for bit, info in ERROR_CODES.items()
    ]
    error_df = pd.DataFrame(error_code_data)
    st.dataframe(error_df, use_container_width=True, hide_index=True)