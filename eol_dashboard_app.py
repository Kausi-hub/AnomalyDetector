import os
import re
import subprocess
from datetime import datetime, timedelta
from collections import defaultdict
import pandas as pd

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =========================================
# CONFIG
# =========================================
TIME_THRESHOLD_SEC = 300

# =========================================
# HELPERS
# =========================================
def extract_unit(filename):
    match = re.search(r"(\d{4,})", filename)
    return match.group(1) if match else "UNKNOWN"

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

# =========================================
# PARSERS
# =========================================
def parse_csv(path):
    start, dur = None, None
    with open(path, 'r', errors='ignore') as f:
        for line in f:
            if "StartMeasDateTime" in line:
                start = datetime.strptime(line.split(",")[1].strip(), "%Y/%m/%d %H:%M:%S")
            if "TotalMeasTime" in line:
                dur = float(line.split(",")[1])
    if start and dur:
        return {
            "file": os.path.basename(path),
            "start": start,
            "end": start + timedelta(minutes=dur),
            "unit": extract_unit(path)
        }

def parse_txt(path):
    start = None
    with open(path, 'r', errors='ignore') as f:
        for line in f:
            if line.lower().startswith("date"):
                raw = line.replace("date", "").strip()
                start = datetime.strptime(raw, "%a %b %d %I:%M:%S.%f %p %Y")
                break

    last = 0
    with open(path, 'r', errors='ignore') as f:
        for line in f:
            m = re.match(r"\s*(\d+\.\d+)", line)
            if m:
                last = float(m.group(1))

    if start:
        return {
            "file": os.path.basename(path),
            "start": start,
            "end": start + timedelta(seconds=last),
            "unit": extract_unit(path)
        }

# =========================================
# CORE ENGINE
# =========================================
def run_analysis(csv_folder, txt_folder):

    def collect(folder, ext):
        return [os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.endswith(ext)]

    csv_runs = [r for f in collect(csv_folder, ".csv") if (r := parse_csv(f))]
    txt_runs = [r for f in collect(txt_folder, ".txt") if (r := parse_txt(f))]

    matches, rows = [], []

    for txt in txt_runs:
        candidates = [c for c in csv_runs if c["unit"] == txt["unit"]] or csv_runs

        best = None
        best_score = -1

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
            "Δ Time (s)": round(diff, 1),
            "Status": status
        })

    df = pd.DataFrame(rows)

    # ---------- GROUP ----------
    groups = defaultdict(lambda: {"csv": [], "txt": []})
    for r in csv_runs: groups[r["unit"]]["csv"].append(r)
    for r in txt_runs: groups[r["unit"]]["txt"].append(r)

    # ---------- FIGURE ----------
    fig = make_subplots(
        rows=2, cols=1,
        specs=[[{"type": "xy"}], [{"type": "domain"}]],
        row_heights=[0.7, 0.3]
    )

    y = 0
    ypos = {}
    labels = []

    for unit in sorted(groups.keys()):
        fig.add_trace(go.Bar(x=[0], y=[y], marker=dict(opacity=0)))
        labels.append(f"UNIT {unit}")
        y += 1

        for run in groups[unit]["csv"]:
            fig.add_trace(go.Bar(
                x=[(run["end"] - run["start"]).total_seconds()],
                y=[y],
                base=run["start"],
                orientation='h',
                marker=dict(color="blue"),
                hovertemplate=f"{run['file']}"
            ))
            ypos[run["file"]] = y
            labels.append(run["file"])
            y += 1

        for run in groups[unit]["txt"]:
            fig.add_trace(go.Bar(
                x=[(run["end"] - run["start"]).total_seconds()],
                y=[y],
                base=run["start"],
                orientation='h',
                marker=dict(color="orange"),
                hovertemplate=f"{run['file']}"
            ))
            ypos[run["file"]] = y
            labels.append(run["file"])
            y += 1

    # link lines
    for m in matches:
        fig.add_trace(go.Scatter(
            x=[m["txt"]["start"], m["csv"]["start"]],
            y=[ypos[m["txt"]["file"]], ypos[m["csv"]["file"]]],
            mode="lines",
            line=dict(color="black", dash="dot"),
            hovertemplate=f"Confidence: {m['confidence']}"
        ))

    # table
    colors = [
        ["#ccffcc" if r["Status"] == "SAME RUN" else "#ffcccc"] * len(df.columns)
        for _, r in df.iterrows()
    ]

    fig.add_trace(go.Table(
        header=dict(values=list(df.columns), fill_color="lightgray"),
        cells=dict(
            values=[df[c] for c in df.columns],
            fill_color=list(map(list, zip(*colors)))
        )
    ), row=2, col=1)

    fig.update_layout(height=900, showlegend=False)

    return fig, df

# =========================================
# DASH UI
# =========================================
app = dash.Dash(__name__)

app.layout = html.Div([

    html.H2("EOL Matching Dashboard"),

    html.Div([
        html.Label("CSV Folder Path"),
        dcc.Input(id="csv-path", type="text", style={"width": "70%"}),
        html.Button("Open", id="open-csv")
    ]),

    html.Br(),

    html.Div([
        html.Label("TXT Folder Path"),
        dcc.Input(id="txt-path", type="text", style={"width": "70%"}),
        html.Button("Open", id="open-txt")
    ]),

    html.Br(),

    html.Button("Run Analysis", id="run"),
    html.Button("Export Report", id="export"),

    dcc.Download(id="download"),
    dcc.Graph(id="graph")

])

# =========================================
# OPEN FOLDER BUTTONS
# =========================================
@app.callback(
    Output("csv-path", "value"),
    Input("open-csv", "n_clicks"),
    State("csv-path", "value"),
    prevent_initial_call=True
)
def open_csv(n, path):
    if path and os.path.exists(path):
        subprocess.Popen(f'explorer "{path}"')
    return path

@app.callback(
    Output("txt-path", "value"),
    Input("open-txt", "n_clicks"),
    State("txt-path", "value"),
    prevent_initial_call=True
)
def open_txt(n, path):
    if path and os.path.exists(path):
        subprocess.Popen(f'explorer "{path}"')
    return path

# =========================================
# RUN
# =========================================
@app.callback(
    Output("graph", "figure"),
    Input("run", "n_clicks"),
    State("csv-path", "value"),
    State("txt-path", "value")
)
def run_dashboard(n, csv_path, txt_path):
    if not n or not os.path.exists(csv_path) or not os.path.exists(txt_path):
        return go.Figure()

    fig, _ = run_analysis(csv_path, txt_path)
    return fig

# =========================================
# EXPORT
# =========================================
@app.callback(
    Output("download", "data"),
    Input("export", "n_clicks"),
    State("csv-path", "value"),
    State("txt-path", "value"),
    prevent_initial_call=True
)
def export(n, csv_path, txt_path):
    _, df = run_analysis(csv_path, txt_path)
    return dcc.send_data_frame(df.to_csv, "matching_report.csv")

# =========================================
# RUN APP
# =========================================
if __name__ == "__main__":
    app.run(debug=True)