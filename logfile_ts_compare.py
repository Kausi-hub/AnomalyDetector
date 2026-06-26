import os
import re
from datetime import datetime, timedelta
from collections import defaultdict
import pandas as pd
from tkinter import Tk, filedialog
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------- FOLDER PICKER ----------
def pick_folder(title):
    root = Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title=title)
    root.destroy()
    return folder

print("Select CSV folder")
CSV_FOLDER = pick_folder("Select CSV Folder")

print("Select TXT folder")
TXT_FOLDER = pick_folder("Select TXT Folder")

# ---------- CONFIG ----------
TIME_THRESHOLD_SEC = 300

# ---------- HELPERS ----------
def extract_unit(filename):
    match = re.search(r"(\d{4,})", filename)
    return match.group(1) if match else "UNKNOWN"

def overlap_score(a_start, a_end, b_start, b_end):
    latest_start = max(a_start, b_start)
    earliest_end = min(a_end, b_end)
    overlap = max(0, (earliest_end - latest_start).total_seconds())
    union = (max(a_end, b_end) - min(a_start, b_start)).total_seconds()
    return overlap / union if union > 0 else 0

def compute_confidence(diff_sec, overlap, unit_match):
    time_score = max(0, 100 - (diff_sec / TIME_THRESHOLD_SEC * 100))
    overlap_score_scaled = overlap * 100
    unit_boost = 1.2 if unit_match else 0.7
    score = (0.5 * time_score + 0.5 * overlap_score_scaled) * unit_boost
    return round(min(100, score), 1)

# ---------- PARSERS ----------
def parse_csv(path):
    start = None
    duration = None

    with open(path, 'r', errors='ignore') as f:
        for line in f:
            if "StartMeasDateTime" in line:
                start = datetime.strptime(line.split(",")[1].strip(), "%Y/%m/%d %H:%M:%S")
            if "TotalMeasTime" in line:
                duration = float(line.split(",")[1])

    if start and duration:
        return {
            "file": os.path.basename(path),
            "start": start,
            "end": start + timedelta(minutes=duration),
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

# ---------- LOAD FILES ----------
def collect(folder, ext):
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(ext)]

csv_runs = [r for f in collect(CSV_FOLDER, ".csv") if (r := parse_csv(f))]
txt_runs = [r for f in collect(TXT_FOLDER, ".txt") if (r := parse_txt(f))]

# ---------- MATCHING ----------
matches = []
report_rows = []

for txt in txt_runs:
    candidates = [c for c in csv_runs if c["unit"] == txt["unit"]] or csv_runs

    comparisons = []

    for csv in candidates:
        diff = abs((txt["start"] - csv["start"]).total_seconds())
        overlap = overlap_score(txt["start"], txt["end"], csv["start"], csv["end"])
        unit_match = csv["unit"] == txt["unit"]

        confidence = compute_confidence(diff, overlap, unit_match)

        comparisons.append({
            "csv": csv,
            "diff": round(diff, 2),
            "overlap": overlap,
            "confidence": confidence
        })

    comparisons.sort(key=lambda x: (-x["overlap"], x["diff"]))
    best = comparisons[0]

    status = "SAME RUN" if best["diff"] <= TIME_THRESHOLD_SEC else "DIFFERENT RUN"

    matches.append({
        "txt": txt,
        "csv": best["csv"],
        "confidence": best["confidence"]
    })

    report_rows.append({
        "TXT File": txt["file"],
        "CSV File": best["csv"]["file"],
        "Δ Time (s)": best["diff"],
        "Status": status
    })

df = pd.DataFrame(report_rows)
df.to_csv("advanced_matching_report.csv", index=False)

# ---------- GROUP BY UNIT ----------
unit_groups = defaultdict(lambda: {"csv": [], "txt": []})

for r in csv_runs:
    unit_groups[r["unit"]]["csv"].append(r)
for r in txt_runs:
    unit_groups[r["unit"]]["txt"].append(r)

sorted_units = sorted(unit_groups.keys())

# ---------- PLOT ----------
fig = make_subplots(
    rows=2, cols=1,
    specs=[[{"type": "xy"}], [{"type": "domain"}]],
    shared_xaxes=True,
    row_heights=[0.7, 0.3],
    vertical_spacing=0.12,
    subplot_titles=("EOL Timeline Alignment (Grouped by Unit)", "Matching Results")
)

y = 0
y_positions = {}
y_labels = []

# ---------- TIMELINE ----------
for unit in sorted_units:

    # Unit label row
    fig.add_trace(go.Bar(x=[0], y=[y], marker=dict(opacity=0), showlegend=False), row=1, col=1)
    y_labels.append(f"UNIT {unit}")
    y += 1

    # CSV runs
    for run in sorted(unit_groups[unit]["csv"], key=lambda x: x["start"]):
        fig.add_trace(go.Bar(
            x=[(run["end"] - run["start"]).total_seconds()],
            y=[y],
            base=run["start"],
            orientation='h',
            marker=dict(color='blue'),
            hovertemplate=f"{run['file']}<br>Unit:{unit}<extra></extra>",
            showlegend=False
        ), row=1, col=1)

        y_positions[run["file"]] = y
        y_labels.append(run["file"])
        y += 1

    # TXT runs
    for run in sorted(unit_groups[unit]["txt"], key=lambda x: x["start"]):
        fig.add_trace(go.Bar(
            x=[(run["end"] - run["start"]).total_seconds()],
            y=[y],
            base=run["start"],
            orientation='h',
            marker=dict(color='orange'),
            hovertemplate=f"{run['file']}<br>Unit:{unit}<extra></extra>",
            showlegend=False
        ), row=1, col=1)

        y_positions[run["file"]] = y
        y_labels.append(run["file"])
        y += 1

    y += 1
    y_labels.append("")

# ---------- LINK LINES ----------
for m in matches:
    txt = m["txt"]
    csv = m["csv"]

    fig.add_trace(go.Scatter(
        x=[txt["start"], csv["start"]],
        y=[y_positions[txt["file"]], y_positions[csv["file"]]],
        mode="lines",
        line=dict(color="black", dash="dot"),
        hovertemplate=f"Confidence: {m['confidence']}",
        showlegend=False
    ), row=1, col=1)

# ---------- Y AXIS ----------
fig.update_yaxes(
    tickmode='array',
    tickvals=list(range(len(y_labels))),
    ticktext=y_labels
)

# ---------- TABLE ----------
colors = [
    ["#ccffcc" if r["Status"] == "SAME RUN" else "#ffcccc"] * len(df.columns)
    for _, r in df.iterrows()
]

fig.add_trace(go.Table(
    header=dict(values=list(df.columns), fill_color="lightgray"),
    cells=dict(
        values=[df[col] for col in df.columns],
        fill_color=list(map(list, zip(*colors))),
        align="center"
    )
), row=2, col=1)

# ---------- FINAL LAYOUT ----------
fig.update_layout(
    height=950,
    title="EOL Matching Dashboard",
    showlegend=False
)

# ---------- EXPORT ----------
fig.write_html("EOL_dashboard.html")

print("✅ Done!")
print("📄 HTML: EOL_dashboard.html")
print("📊 CSV: advanced_matching_report.csv")