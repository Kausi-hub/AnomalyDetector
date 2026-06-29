import re
from datetime import datetime, timedelta
from collections import defaultdict, Counter

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

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

def normalize_name(name):
    name = name.lower()
    name = re.sub(r".*::", "", name)
    name = re.sub(r"^v_|^measured|^rig_", "", name)
    name = re.sub(r"[_\s]+", "", name)
    return name

def categorize_signal(name):
    n = name.lower()
    if "speed" in n:
        return "Speed"
    elif "torque" in n:
        return "Torque"
    elif "temp" in n:
        return "Temperature"
    elif "volt" in n:
        return "Voltage"
    return "Other"

def confidence_label(val):
    if val >= 75:
        return "🟢"
    elif val >= 40:
        return "🟡"
    else:
        return "🔴"

# ==============================
# CSV PARSER
# ==============================
def parse_csv(file):
    lines = file.getvalue().decode(errors="ignore").splitlines()

    start, dur = None, None
    headers, data = [], []
    reading = False

    for line in lines:
        if "StartMeasDateTime" in line:
            start = datetime.strptime(line.split(",")[1].strip(), "%Y/%m/%d %H:%M:%S")

        if "TotalMeasTime" in line:
            dur = float(line.split(",")[1])

        if line.startswith("Time,"):
            headers = [h.strip() for h in line.split(",")]
            reading = True
            continue

        if reading and re.match(r"\s*\d+\.\d+", line):
            data.append(line.split(","))

    if not start or not dur or not headers or not data:
        return None

    clean = []
    for r in data:
        if len(r) < len(headers):
            r += [None] * (len(headers) - len(r))
        elif len(r) > len(headers):
            r = r[:len(headers)]
        clean.append(r)

    df = pd.DataFrame(clean, columns=headers).apply(pd.to_numeric, errors="coerce")

    signals = {
        normalize_name(col): {"values": df[col].dropna().values}
        for col in df.columns
    }

    return {
        "file": file.name,
        "start": start,
        "end": start + timedelta(minutes=dur),
        "unit": extract_unit(file.name),
        "signals": signals
    }

# ==============================
# TXT PARSER
# ==============================
def parse_txt(file):
    lines = file.getvalue().decode(errors="ignore").splitlines()

    start, last = None, 0
    signals = defaultdict(list)
    messages = []

    for line in lines:

        if line.lower().startswith("date"):
            try:
                start = datetime.strptime(
                    line.replace("date", "").strip(),
                    "%a %b %d %I:%M:%S.%f %p %Y"
                )
            except:
                pass

        t = re.match(r"\s*(\d+\.\d+)", line)
        if t:
            last = float(t.group(1))

        m = re.search(r"::([\w]+)\s*=\s*([-\d\.]+)", line)
        if m:
            signals[normalize_name(m.group(1))].append(float(m.group(2)))

        m = re.search(r"CANFD.*Tx\s+([0-9A-Fa-f]+)", line)
        if m:
            messages.append(m.group(1))

    if not start:
        return None

    return {
        "file": file.name,
        "start": start,
        "end": start + timedelta(seconds=last),
        "unit": extract_unit(file.name),
        "signals": signals,
        "messages": messages
    }

# ==============================
# MATCHING
# ==============================
def find_pairs(csv, txt):
    return [(c, t) for c in csv for t in txt if c in t or t in c]

def signal_similarity(csv, txt, pairs):
    scores = []
    for c, t in pairs:
        s1 = pd.Series(csv[c]["values"])
        s2 = pd.Series(txt[t])
        if len(s1) < 10 or len(s2) < 10:
            continue
        m = min(len(s1), len(s2))
        corr = s1[:m].corr(s2[:m])
        if pd.notna(corr):
            scores.append(max(0, corr))
    return round((sum(scores) / len(scores)) * 100, 1) if scores else 0

def msg_similarity(m1, m2):
    c1, c2 = Counter(m1), Counter(m2)
    common = set(c1) & set(c2)
    if not common:
        return 0
    overlap = sum(min(c1[k], c2[k]) for k in common)
    total = sum(c1.values()) + sum(c2.values())
    return (2 * overlap / total) * 100

def best_signals(csv, txt, pairs):
    scored = []
    for c, t in pairs:
        s1 = pd.Series(csv[c]["values"])
        s2 = pd.Series(txt[t])
        if len(s1) < 10 or len(s2) < 10:
            continue
        m = min(len(s1), len(s2))
        corr = s1[:m].corr(s2[:m])
        if pd.notna(corr):
            scored.append((c, t, abs(corr)))
    scored.sort(key=lambda x: x[2], reverse=True)
    return [(c, t) for c, t, _ in scored[:3]]

def confidence(diff, sig, msg):
    return round(min(100, (0.4 * (100 - diff/300*100) + 0.4 * sig + 0.2 * msg)), 1)

# ==============================
# ANALYSIS
# ==============================
def run(csv_runs, txt_runs):
    rows, matches = [], []

    for txt in txt_runs:
        best = None
        best_score = -1

        for csv in csv_runs:

            if not csv.get("start") or not txt.get("start"):
                continue

            diff = abs((txt["start"] - csv["start"]).total_seconds())
            pairs = find_pairs(csv["signals"], txt["signals"])
            sig = signal_similarity(csv["signals"], txt["signals"], pairs)
            msg = msg_similarity(txt["messages"], [])
            conf = confidence(diff, sig, msg)

            if best is None or conf > best_score:
                best_score = conf
                best = (csv, diff, conf, sig, msg, pairs)

        if best is None:
            continue

        csv, diff, conf, sig, msg, pairs = best

        status = (
            "🟢 SAME RUN" if conf >= 75 else
            "🟡 POSSIBLE MATCH" if conf >= 40 else
            "🔴 DIFFERENT RUN"
        )

        matches.append({"csv": csv, "txt": txt, "pairs": pairs})

        rows.append({
            "TXT File": txt["file"],
            "CSV File": csv["file"],
            "Δ Time (s)": round(diff, 1),
            "Signal Similarity": sig,
            "Message Similarity": round(msg, 1),
            "Confidence": conf,
            "Status": status
        })

    return pd.DataFrame(rows), matches

# ==============================
# TIMELINE
# ==============================
def timeline(csv_runs, txt_runs, matches):
    fig = go.Figure()
    y = 0
    pos = {}

    for r in csv_runs:
        fig.add_bar(x=[(r["end"]-r["start"]).seconds], y=[y],
                    base=r["start"], orientation='h', marker=dict(color="blue"))
        pos[r["file"]] = y; y += 1

    for r in txt_runs:
        fig.add_bar(x=[(r["end"]-r["start"]).seconds], y=[y],
                    base=r["start"], orientation='h', marker=dict(color="orange"))
        pos[r["file"]] = y; y += 1

    for m in matches:
        fig.add_trace(go.Scatter(
            x=[m["txt"]["start"], m["csv"]["start"]],
            y=[pos[m["txt"]["file"]], pos[m["csv"]["file"]]],
            mode="lines",
            line=dict(dash="dot", color="gray")
        ))

    return fig

# ==============================
# UI
# ==============================
st.title("EOL Matching Dashboard")

st.info("Compare CSV and TXT logs using time, signals, and CAN patterns")

with st.expander("ℹ️ How to Use This App"):
    st.markdown("""
- Upload CSV & TXT files  
- Run analysis  
- Review match table, timeline & signals  

🟢 Strong ≥75 | 🟡 Partial 40–74 | 🔴 Weak <40  
""")

csv_files = st.file_uploader("Upload CSV", accept_multiple_files=True)
txt_files = st.file_uploader("Upload TXT", accept_multiple_files=True)

if st.button("Run Analysis"):

    with st.spinner("Analyzing..."):

        csv_runs = [r for f in csv_files if (r := parse_csv(f))]
        txt_runs = [r for f in txt_files if (r := parse_txt(f))]

        if not csv_runs or not txt_runs:
            st.error("No valid data")
            st.stop()

        df, matches = run(csv_runs, txt_runs)

        st.session_state.update({
            "df": df,
            "matches": matches,
            "csv": csv_runs,
            "txt": txt_runs
        })

# ==============================
# DISPLAY
# ==============================
if "df" in st.session_state:

    df = st.session_state["df"]
    matches = st.session_state["matches"]

    # ✅ Cloud-safe RYG display
    display_df = df.copy()
    display_df["Match"] = display_df["Confidence"].apply(confidence_label)

    cols = ["Match"] + [c for c in display_df.columns if c != "Match"]
    display_df = display_df[cols]

    st.subheader("📋 Results")
    st.dataframe(display_df, use_container_width=True)

    st.caption("🟢 Strong | 🟡 Partial | 🔴 Weak")

    st.subheader("📊 Timeline")
    st.plotly_chart(timeline(
        st.session_state["csv"],
        st.session_state["txt"],
        matches
    ))

    st.caption("🔵 CSV | 🟠 TXT | ⋯ Matches")

    idx = st.selectbox("Select Match", range(len(df)))
    m = matches[idx]

    cat = st.selectbox("Signal Filter",
                       ["All","Speed","Torque","Temperature","Voltage","Other"])

    filtered = [(c, t) for c, t in m["pairs"]
                if cat == "All" or categorize_signal(c) == cat]

    best = best_signals(m["csv"]["signals"], m["txt"]["signals"], filtered)

    options = [f"{c} ↔ {t}" for c, t in filtered]

    selected = st.multiselect("Signals", options,
                             default=[f"{c} ↔ {t}" for c, t in best])

    fig = go.Figure()

    for s in selected:
        c, t = s.split(" ↔ ")
        a = m["csv"]["signals"][c]["values"]
        b = m["txt"]["signals"][t]

        L = min(len(a), len(b))
        x = list(range(L))

        fig.add_trace(go.Scatter(x=x, y=a[:L], name=f"CSV {c}"))
        fig.add_trace(go.Scatter(x=x, y=b[:L], name=f"TXT {t}",
                                 line=dict(dash="dot")))

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🧠 Explanation")
    row = df.iloc[idx]
    st.write(f"Time diff: {row['Δ Time (s)']} sec")
    st.write(f"Signal similarity: {row['Signal Similarity']}%")
    st.write(f"Message similarity: {row['Message Similarity']}%")
    st.write(f"Confidence: {row['Confidence']}%")