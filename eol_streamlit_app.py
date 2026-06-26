import os
import re
from datetime import datetime, timedelta
from collections import defaultdict
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==============================
# CONFIG
# ==============================
TIME_THRESHOLD_SEC = 300

# ==============================
# HELPERS
# ==============================
def extract_unit(name):
    m = re.search(r"(\d{4,})", name)
    return m.group(1) if m else "UNKNOWN"

def overlap_score(a_start, a_end, b_start, b_end):
    latest = max(a_start, b_start)
    earliest = min(a_end, b_end)
    overlap = max(0, (earliest - latest).total_seconds())
    union = (max(a_end, b_end) - min(a_start, b_start)).total_seconds()
    return overlap / union if union else 0

def compute_confidence(diff, overlap, unit_match):
    time_score = max(0, 100 - (diff / TIME_THRESHOLD_SEC * 100))
    overlap_scaled = overlap * 100
    boost = 1.2 if unit_match else 0.7
    return round(min(100, (0.5*time_score + 0.5*overlap_scaled)*boost), 1)

# ==============================
# PARSERS
# ==============================
def parse_csv(file):
    lines = file.getvalue().decode(errors="ignore").splitlines()
    start, dur = None, None

    for line in lines:
        if "StartMeasDateTime" in line:
            start = datetime.strptime(line.split(",")[1].strip(), "%Y/%m/%d %H:%M:%S")
        if "TotalMeasTime" in line:
            dur = float(line.split(",")[1])

    if start and dur:
        return {
            "file": file.name,
            "start": start,
            "end": start + timedelta(minutes=dur),
            "unit": extract_unit(file.name)
        }

def parse_txt(file):
    lines = file.getvalue().decode(errors="ignore").splitlines()
    start, last = None, 0

    for line in lines:
        if line.lower().startswith("date"):
            raw = line.replace("date", "").strip()
            start = datetime.strptime(raw, "%a %b %d %I:%M:%S.%f %p %Y")

        m = re.match(r"\s*(\d+\.\d+)", line)
        if m:
            last = float(m.group(1))

    if start:
        return {
            "file": file.name,
            "start": start,
            "end": start + timedelta(seconds=last),
            "unit": extract_unit(file.name)
        }

# ==============================
# ANALYSIS
# ==============================
def run_analysis(csv_runs, txt_runs):

    matches, rows = [], []

    for txt in txt_runs:
        candidates = [c for c in csv_runs if c["unit"] == txt["unit"]] or csv_runs

        best, best_score = None, -1

        for csv in candidates:
            diff = abs((txt["start"] - csv["start"]).total_seconds())
            overlap = overlap_score(txt["start"], txt["end"], csv["start"], csv["end"])
            conf = compute_confidence(diff, overlap, csv["unit"] == txt["unit"])

            if conf > best_score:
                best_score = conf
                best = (csv, diff, conf)

        csv, diff, conf = best
        status = "SAME RUN" if diff <= TIME_THRESHOLD_SEC else "DIFFERENT RUN"

        matches.append({"txt": txt, "csv": csv, "confidence": conf})

        rows.append({
            "TXT File": txt["file"],
            "CSV File": csv["file"],
            "Δ Time (s)": round(diff,1),
            "Status": status
        })

    df = pd.DataFrame(rows)

    # ---------- GROUP ----------
    groups = defaultdict(lambda: {"csv":[], "txt":[]})
    for r in csv_runs: groups[r["unit"]]["csv"].append(r)
    for r in txt_runs: groups[r["unit"]]["txt"].append(r)

    fig = make_subplots(
        rows=2, cols=1,
        specs=[[{"type":"xy"}],[{"type":"domain"}]],
        row_heights=[0.7,0.3]
    )

    y, ypos = 0, {}

    for unit in sorted(groups.keys()):
        fig.add_trace(go.Bar(x=[0], y=[y], marker=dict(opacity=0)))
        y += 1

        for run in groups[unit]["csv"]:
            fig.add_trace(go.Bar(
                x=[(run["end"]-run["start"]).total_seconds()],
                y=[y], base=run["start"],
                orientation='h',
                marker=dict(color="blue")
            ))
            ypos[run["file"]] = y
            y += 1

        for run in groups[unit]["txt"]:
            fig.add_trace(go.Bar(
                x=[(run["end"]-run["start"]).total_seconds()],
                y=[y], base=run["start"],
                orientation='h',
                marker=dict(color="orange")
            ))
            ypos[run["file"]] = y
            y += 1

    for m in matches:
        fig.add_trace(go.Scatter(
            x=[m["txt"]["start"], m["csv"]["start"]],
            y=[ypos[m["txt"]["file"]], ypos[m["csv"]["file"]]],
            mode="lines",
            line=dict(dash="dot")
        ))

    fig.update_layout(height=800, showlegend=False)

    return fig, df

# ==============================
# UI
# ==============================
st.title("EOL Matching Dashboard")

csv_files = st.file_uploader("Upload CSV files", accept_multiple_files=True)
txt_files = st.file_uploader("Upload TXT files", accept_multiple_files=True)

if st.button("Run Analysis"):

    csv_runs = [parse_csv(f) for f in csv_files if parse_csv(f)]
    txt_runs = [parse_txt(f) for f in txt_files if parse_txt(f)]

    fig, df = run_analysis(csv_runs, txt_runs)

    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df)

    st.download_button(
        "Download Report",
        df.to_csv(index=False),
        file_name="matching_report.csv"
    )