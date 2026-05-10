# Main Streamlit application for Anomaly Detection Dashboard

import streamlit as st
import numpy as np
import pandas as pd

from config import HIGH_THRESHOLD_DEFAULT, WARN_THRESHOLD_DEFAULT, TOP_ROOT_CAUSES_DEFAULT
from ui_components import (
    setup_page_config, display_header, create_sidebar, parse_logs_cached,
    display_log_metadata, display_kpi_metrics, create_main_tabs,
    display_divergence_matrix_tab, display_root_cause_tab, display_heatmap_tab,
    display_signal_details_tab, display_log_model_tab, display_error_reference_tab
)
from ml_models import build_reference_profile, train_pass_iforest_model, compare_to_pass_model


def main():
    """Main application entry point."""
    # Setup page configuration
    setup_page_config()

    # Display header
    display_header()

    # Initialize sidebar variables
    passed_files = None
    failed_files = None
    high_threshold = HIGH_THRESHOLD_DEFAULT
    warn_threshold = WARN_THRESHOLD_DEFAULT
    top_root_causes = TOP_ROOT_CAUSES_DEFAULT
    signal_filter = ""

    # Create sidebar and get user inputs
    passed_files, failed_files, high_threshold, warn_threshold, top_root_causes, signal_filter = create_sidebar(
        passed_files, failed_files, high_threshold, warn_threshold, top_root_causes, signal_filter
    )

    # Check if files are uploaded
    if not passed_files or not failed_files:
        st.info("Use the file browser in the left sidebar to upload at least one passed log and one failed log.")
        st.stop()

    # Parse uploaded files with caching
    with st.spinner("Parsing logs..."):
        passed_logs = parse_logs_cached(passed_files, "passed")
        failed_logs = parse_logs_cached(failed_files, "failed")

    all_logs = passed_logs + failed_logs

    # Display log metadata
    display_log_metadata(all_logs)

    # Build reference profile and train ML model
    ref_profile = build_reference_profile(passed_logs)

    # Cache the trained model
    if "pass_model" not in st.session_state:
        st.session_state["pass_model"] = train_pass_iforest_model(passed_logs)
    pass_model = st.session_state["pass_model"]

    # Compare failed logs to reference
    all_divergence = []
    for log in failed_logs:
        divergence = compare_to_pass_model(log, ref_profile, pass_model)
        if not divergence.empty and signal_filter:
            divergence = divergence[divergence["signal"].str.contains(signal_filter, case=False, na=False)]
        if not divergence.empty:
            all_divergence.append(divergence)

    if not all_divergence:
        st.warning("No comparable signals were found. Check that passed and failed logs use similar signal naming.")
        st.stop()

    divergence_df = pd.concat(all_divergence, ignore_index=True)

    # Apply user-defined thresholds
    divergence_df["status"] = np.where(
        divergence_df["divergence_score"] >= high_threshold, "❌",
        np.where(divergence_df["divergence_score"] >= warn_threshold, "⚠️", "✅")
    )
    divergence_df["severity"] = np.where(
        divergence_df["divergence_score"] >= high_threshold, "High",
        np.where(divergence_df["divergence_score"] >= warn_threshold, "Medium", "Low")
    )

    # Display KPI metrics
    display_kpi_metrics(failed_logs, divergence_df)

    # Create main content tabs
    tab_matrix, tab_root, tab_heatmap, tab_details, tab_model, tab_error = create_main_tabs()

    # Display tab contents
    with tab_matrix:
        display_divergence_matrix_tab(divergence_df)

    with tab_root:
        display_root_cause_tab(divergence_df, top_root_causes)

    with tab_heatmap:
        display_heatmap_tab(divergence_df)

    with tab_details:
        display_signal_details_tab(divergence_df)

    with tab_model:
        display_log_model_tab(all_logs)

    with tab_error:
        display_error_reference_tab()


if __name__ == "__main__":
    main()