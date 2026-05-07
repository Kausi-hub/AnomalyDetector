import re
import io
import math
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest


# ============================================================
# AWD EOL Error Code Dictionary
# Based on provided EOL AWD error-code matrix image.
# ============================================================

ERROR_CODES = {
    0: {
        "status_value": "2^0",
        "name": "ETM Error Status detected",
        "description": "ETM Error Status detected; check V_ETM_Disturbance_Status for more error information.",
        "root_hint": "ETM internal diagnostic or disturbance status."
    },
    1: {
        "status_value": "2^1",
        "name": "PTU Error Status detected",
        "description": "PTU Error Status detected; check V_PTU_Disturbance_Status for more error information.",
        "root_hint": "PTU internal diagnostic or disturbance status."
    },
    2: {
        "status_value": "2^2",
        "name": "Emergency stop",
        "description": "Emergency Stop triggered by rig operator.",
        "root_hint": "Operator stop / rig safety condition."
    },
    3: {
        "status_value": "2^3",
        "name": "Maximum torque exceeded",
        "description": "Overall torque observation. Error set if measured torque exceeds Diag_Measured_Tq_Limit.",
        "root_hint": "Measured torque over limit."
    },
    4: {
        "status_value": "2^4",
        "name": "Maximum clutch temperature exceeded",
        "description": "Thermal protection. Error set if calculated temperature exceeds Diag_Clutch_Temp_Limit.",
        "root_hint": "Thermal / clutch temperature limit."
    },
    5: {
        "status_value": "2^5",
        "name": "Initial temperature condition NOT fulfilled at EOL-Start",
        "description": "Check ambient temperature condition before EOL starts.",
        "root_hint": "Ambient / initial temperature precondition."
    },
    6: {
        "status_value": "2^6",
        "name": "Dock in side shafts error",
        "description": "Side shaft docking has not completed before timeout.",
        "root_hint": "Mechanical docking / shaft engagement timeout."
    },
    7: {
        "status_value": "2^7",
        "name": "Run in Error",
        "description": "Accumulated energy during Run-in stage is below configured minimum.",
        "root_hint": "Run-in energy too low; missing motion, torque, or clutch energy."
    },
    8: {
        "status_value": "2^8",
        "name": "CWO_OOR_at_CWO",
        "description": "Learned clutch wear-off position out of acceptable range.",
        "root_hint": "Clutch wear-off learn value out of range."
    },
    9: {
        "status_value": "2^9",
        "name": "CWO_Delta_OOR_at_CWO",
        "description": "Delta between learned clutch wear-off position and ETM position at zero torque exceeds threshold.",
        "root_hint": "Clutch/ETM zero torque offset excessive."
    },
    10: {
        "status_value": "2^10",
        "name": "Maximum Classification Value Exceeded",
        "description": "At least one classification value exceeded allowed range.",
        "root_hint": "Classification learned value over limit."
    },
    11: {
        "status_value": "2^11",
        "name": "Torque tolerance exceeded at Verification",
        "description": "Torque tolerance exceeded for at least one verification torque value.",
        "root_hint": "Torque verification mismatch."
    },
    12: {
        "status_value": "2^12",
        "name": "Power supply voltage condition NOT fulfilled",
        "description": "Check power supply condition during EOL test.",
        "root_hint": "Power supply / voltage precondition."
    },
    13: {
        "status_value": "2^13",
        "name": "Interrupted communication detected",
        "description": "Disturbance on communication detected.",
        "root_hint": "CAN/CAN-FD/network interruption."
    },
    14: {
        "status_value": "2^14",
        "name": "Device Control Request Rejected",
        "description": "ECU rejected a request from Rig-EOL-SW for a test mode or diagnostic service request.",
        "root_hint": "Diagnostic request rejected / negative response."
    },
    15: {
        "status_value": "2^15",
        "name": "Classification Write Request Rejected",
        "description": "ECU rejected a request from Rig-EOL-SW to write classification values.",
        "root_hint": "NVM/write request rejected."
    },
    16: {
        "status_value": "2^16",
        "name": "ECU Initialization Error",
        "description": "ECU Master Operation State or ETM Operation State did not achieve expected RUN state.",
        "root_hint": "ECU/ETM operation state not RUN."
    },
    17: {
        "status_value": "2^17",
        "name": "Classification Values Don’t match between Rig and ECU",
        "description": "Classification values in Rig-EOL-SW do not match ECU EEPROM classification values.",
        "root_hint": "Rig/ECU calibration or classification mismatch."
    },
    18: {
        "status_value": "2^18",
        "name": "Maximum Allowed Open Loop Torque Exceeded",
        "description": "Open-loop commanded torque difference exceeds allowed classification torque value.",
        "root_hint": "Open-loop torque too high."
    },
    19: {
        "status_value": "2^19",
        "name": "Step Response not fulfilled",
        "description": "First step verification response time is greater than target response time.",
        "root_hint": "Dynamic response too slow."
    },
    20: {
        "status_value": "2^20",
        "name": "Rig Motor Speed Error",
        "description": "Measured rig motor speed inaccurate compared with commanded motor speed.",
        "root_hint": "Rig motor speed / command tracking issue."
    },
    21: {
        "status_value": "2^21",
        "name": "Precondition not satisfied",
        "description": "Any precondition is not satisfied for any step.",
        "root_hint": "Generic missing test precondition."
    },
    22: {
        "status_value": "2^22",
        "name": "CWOD Failed Calibration",
        "description": "Calibration request indicates one of the calibration conditions was not met.",
        "root_hint": "CWOD calibration condition not met."
    },
    23: {
        "status_value": "2^23",
        "name": "DTC Set in ECU",
        "description": "ECU has reported Diagnostic Trouble Code during/after EOL test.",
        "root_hint": "ECU DTC / wiring / harness / clutch / CAN bus issue."
    },
    24: {
        "status_value": "2^24",
        "name": "EEPROM Read Value Incorrect",
        "description": "EEPROM values read after shutdown did not match expected written values.",
        "root_hint": "EEPROM write/readback mismatch."
    },
}


# ============================================================
# Parsing
# ============================================================

@dataclass
class ParsedLog:
    name: str
    label: str
    raw_text: str
    events: pd.DataFrame
    features: pd.DataFrame
    metadata: Dict[str, str]


def read_uploaded_file(uploaded_file) -> str:
    data = uploaded_file.read()
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def normalize_signal_name(name: str) -> str:
    name = name.strip()
    name = name.replace("\\_", "_")
    name = name.replace("::", ".")
    name = re.sub(r"\s+", "_", name)
    return name


def parse_value(value_str: str):
    value_str = value_str.strip().strip('"')
    if value_str == "":
        return np.nan

    # Hex-ish payloads, arrays, text statuses remain strings.
    if "[" in value_str or "]" in value_str:
        return value_str

    # Numeric conversion.
    try:
        if re.match(r"^-?\d+(\.\d+)?$", value_str):
            return float(value_str)
    except Exception:
        pass

    # Common ready/status text.
    return value_str


def parse_log_text(text: str, name: str, label: str) -> ParsedLog:
    """
    Supports common patterns in the attached CANoe-style text logs:
      0.000000 Signal := value
      0.000000 SV: ... ::Namespace::Signal = value
      0.020091 CAN 1 Status:chip status error active
      0.079818 CANFD 1 Tx ...
    """
    rows = []
    metadata = {}

    version_match = re.search(r"//\s*version\s+([^\n\r]+)", text)
    uuid_match = re.search(r"//\s*Measurement UUID:\s*([^\n\r]+)", text)
    date_match = re.search(r"^date\s+(.+)$", text, flags=re.MULTILINE)

    if version_match:
        metadata["version"] = version_match.group(1).strip()
    if uuid_match:
        metadata["measurement_uuid"] = uuid_match.group(1).strip()
    if date_match:
        metadata["date"] = date_match.group(1).strip()

    line_pattern = re.compile(r"^\s*(\d+\.\d+)\s+(.*)$")
    assign_pattern = re.compile(r"^([A-Za-z0-9_:\\.\-\[\]/]+)\s*:=\s*(.*)$")
    sv_pattern = re.compile(r"^SV:\s+.*?(::[A-Za-z0-9_:]+)\s*=\s*(.*)$")
    can_status_pattern = re.compile(r"^(CAN(?:FD)?\s+\d+)\s+Status:(.*)$", re.IGNORECASE)
    can_tx_pattern = re.compile(r"^(CANFD|CAN)\s+(\d+)\s+(Tx|Rx)\s+([0-9A-Fa-fx]+)\s+([A-Za-z0-9_]+)?\s*(.*)$")

    for line in text.splitlines():
        m = line_pattern.match(line)
        if not m:
            continue

        t = float(m.group(1))
        payload = m.group(2).strip()

        sv = sv_pattern.match(payload)
        if sv:
            signal = normalize_signal_name(sv.group(1).lstrip(":"))
            value = parse_value(sv.group(2))
            rows.append({
                "time": t, "signal": signal, "value": value,
                "event_type": "SV", "raw": line
            })
            continue

        assign = assign_pattern.match(payload)
        if assign:
            signal = normalize_signal_name(assign.group(1))
            value = parse_value(assign.group(2))
            rows.append({
                "time": t, "signal": signal, "value": value,
                "event_type": "ASSIGN", "raw": line
            })
            continue

        can_status = can_status_pattern.match(payload)
        if can_status:
            signal = normalize_signal_name(can_status.group(1) + "_Status")
            value = can_status.group(2).strip()
            rows.append({
                "time": t, "signal": signal, "value": value,
                "event_type": "CAN_STATUS", "raw": line
            })
            continue

        can_tx = can_tx_pattern.match(payload)
        if can_tx:
            bus_type, channel, direction, can_id, pdu, rest = can_tx.groups()
            signal = normalize_signal_name(f"{bus_type}{channel}_{direction}_{pdu or can_id}")
            rows.append({
                "time": t, "signal": signal, "value": 1.0,
                "event_type": "CAN_FRAME", "raw": line
            })
            continue

    events = pd.DataFrame(rows)

    if events.empty:
        features = pd.DataFrame()
    else:
        events["log_name"] = name
        events["label"] = label
        features = compute_features(events)

    return ParsedLog(name=name, label=label, raw_text=text, events=events, features=features, metadata=metadata)


def to_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def compute_features(events: pd.DataFrame) -> pd.DataFrame:
    feature_rows = []

    for signal, grp in events.groupby("signal"):
        grp = grp.sort_values("time")
        values_raw = grp["value"]
        values_num = to_numeric_series(values_raw)

        valid_num = values_num.dropna()
        n = len(grp)
        n_num = len(valid_num)

        if n_num > 0:
            first = valid_num.iloc[0]
            last = valid_num.iloc[-1]
            mean = valid_num.mean()
            std = valid_num.std(ddof=0) if n_num > 1 else 0.0
            min_v = valid_num.min()
            max_v = valid_num.max()
            span = max_v - min_v
            zero_frac = float((valid_num == 0).mean())
            one_frac = float((valid_num == 1).mean())
            nonzero_frac = float((valid_num != 0).mean())
        else:
            first = last = mean = std = min_v = max_v = span = np.nan
            zero_frac = one_frac = nonzero_frac = np.nan

        transitions = 0
        try:
            transitions = int((values_raw.astype(str).shift() != values_raw.astype(str)).sum() - 1)
            transitions = max(transitions, 0)
        except Exception:
            transitions = 0

        duration = grp["time"].max() - grp["time"].min() if n > 1 else 0.0
        rate = n / duration if duration > 0 else np.nan

        feature_rows.append({
            "signal": signal,
            "count": n,
            "numeric_count": n_num,
            "first": first,
            "last": last,
            "mean": mean,
            "std": std,
            "min": min_v,
            "max": max_v,
            "span": span,
            "zero_frac": zero_frac,
            "one_frac": one_frac,
            "nonzero_frac": nonzero_frac,
            "unique_count": values_raw.astype(str).nunique(),
            "transitions": transitions,
            "duration": duration,
            "event_rate_hz": rate,
            "first_seen": grp["time"].min(),
            "last_seen": grp["time"].max(),
        })

    return pd.DataFrame(feature_rows)


# ============================================================
# Reference profile and anomaly scoring
# ============================================================

FEATURE_COLS = [
    "count", "numeric_count", "last", "mean", "std", "min", "max", "span",
    "zero_frac", "one_frac", "nonzero_frac", "unique_count",
    "transitions", "duration", "event_rate_hz", "first_seen", "last_seen"
]


def build_reference_profile(passed_logs: List[ParsedLog]) -> pd.DataFrame:
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
                mad = np.median(np.abs(vals - median))
                row[f"{col}_ref_median"] = median
                row[f"{col}_ref_mad"] = mad
                row[f"{col}_ref_min"] = vals.min()
                row[f"{col}_ref_max"] = vals.max()
        profile_rows.append(row)

    return pd.DataFrame(profile_rows)


def robust_z(value, median, mad):
    if pd.isna(value) or pd.isna(median):
        return np.nan
    scale = 1.4826 * mad
    if pd.isna(scale) or scale < 1e-9:
        return 0.0 if abs(value - median) < 1e-9 else abs(value - median)
    return abs(value - median) / scale


def compare_to_reference(log: ParsedLog, ref: pd.DataFrame) -> pd.DataFrame:
    if log.features.empty or ref.empty:
        return pd.DataFrame()

    merged = log.features.merge(ref, on="signal", how="outer", indicator=True)

    rows = []
    for _, r in merged.iterrows():
        signal = r["signal"]
        missing_in_failed = r["_merge"] == "right_only"
        new_in_failed = r["_merge"] == "left_only"

        scores = []
        reasons = []

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

        for col in FEATURE_COLS:
            val = pd.to_numeric(pd.Series([r.get(col)]), errors="coerce").iloc[0]
            med = r.get(f"{col}_ref_median", np.nan)
            mad = r.get(f"{col}_ref_mad", np.nan)
            z = robust_z(val, med, mad)
            if not pd.isna(z):
                scores.append(min(z, 25.0))

                if z >= 6:
                    reasons.append(f"{col} deviates strongly")
                elif z >= 3:
                    reasons.append(f"{col} deviates")

        score = float(np.nanmax(scores)) if scores else 0.0

        # Rule-based boosts for common AWD EOL symptoms.
        rule_boost, rule_reason = rule_based_boost(signal, r)
        score = min(100.0, score * 4.0 + rule_boost)

        reason = "; ".join(sorted(set(reasons + rule_reason)))
        if not reason:
            reason = "Matches passed reference behavior"

        if score >= 60:
            status = "❌"
            severity = "High"
        elif score >= 25:
            status = "⚠️"
            severity = "Medium"
        else:
            status = "✅"
            severity = "Low"

        rows.append({
            "log_name": log.name,
            "signal": signal,
            "divergence_score": score,
            "status": status,
            "reason": reason,
            "severity": severity
        })

    return pd.DataFrame(rows)


def rule_based_boost(signal: str, row: pd.Series) -> Tuple[float, List[str]]:
    boost = 0.0
    reasons = []

    last = pd.to_numeric(pd.Series([row.get("last")]), errors="coerce").iloc[0]
    mean = pd.to_numeric(pd.Series([row.get("mean")]), errors="coerce").iloc[0]
    span = pd.to_numeric(pd.Series([row.get("span")]), errors="coerce").iloc[0]
    zero_frac = pd.to_numeric(pd.Series([row.get("zero_frac")]), errors="coerce").iloc[0]
    one_frac = pd.to_numeric(pd.Series([row.get("one_frac")]), errors="coerce").iloc[0]
    transitions = pd.to_numeric(pd.Series([row.get("transitions")]), errors="coerce").iloc[0]

    s = signal.lower()

    if "_inv" in s and not pd.isna(one_frac) and one_frac > 0.8:
        boost += 45
        reasons.append("inverted/auth invalid flag is persistently high")

    if "angvelauth" in s and not pd.isna(zero_frac) and zero_frac > 0.8:
        boost += 25
        reasons.append("wheel angular velocity auth remains invalid/zero")

    if ("flspeed" in s or "frspeed" in s or "speed" in s) and not pd.isna(zero_frac) and zero_frac > 0.95:
        boost += 15
        reasons.append("speed remains zero/static")

    if "gkn_hil.panel" in s and not pd.isna(zero_frac) and zero_frac > 0.95:
        boost += 10
        reasons.append("HIL motion input remains static zero")

    if "ign_txing_rbs_msgs" in s and not pd.isna(transitions) and transitions > 10:
        boost += 20
        reasons.append("RBS message transmission is unstable/toggling")

    if "can" in s and "status" in s:
        raw_value = str(row.get("last", ""))
        if "error" in raw_value.lower():
            boost += 35
            reasons.append("CAN status reports error active")

    if not pd.isna(span) and span == 0 and not pd.isna(zero_frac) and zero_frac > 0.95:
        boost += 5
        reasons.append("signal is stuck at zero")

    return boost, reasons


# ============================================================
# Root-cause rules
# ============================================================

ROOT_CAUSE_RULES = [
    {
        "pattern": ["wheel", "_inv"],
        "error_bits": [13, 21, 23],
        "root_cause": "Wheel-speed validity/authentication issue or missing vehicle motion input",
        "interpretation": "Wheel-speed inverted/auth invalid flags diverge from passed reference."
    },
    {
        "pattern": ["angvelauth"],
        "error_bits": [13, 21, 23],
        "root_cause": "Wheel-speed authentication not valid",
        "interpretation": "Wheel angular velocity authorization remains invalid or zero."
    },
    {
        "pattern": ["ign_txing_rbs_msgs"],
        "error_bits": [13, 21],
        "root_cause": "RBS transmit gating instability",
        "interpretation": "RBS messages drop/toggle relative to passed reference; often symptom, not root cause."
    },
    {
        "pattern": ["can", "status"],
        "error_bits": [13, 23],
        "root_cause": "CAN/CAN-FD communication disturbance",
        "interpretation": "CAN controller or network status reports error active."
    },
    {
        "pattern": ["gkn_hil.panel"],
        "error_bits": [7, 20, 21],
        "root_cause": "Rig/HIL motion inputs are static or missing",
        "interpretation": "Panel acceleration/yaw/steering inputs remain static; may explain run-in energy failure."
    },
    {
        "pattern": ["diagnostic", "negative"],
        "error_bits": [14],
        "root_cause": "ECU rejected diagnostic/device-control request",
        "interpretation": "Diagnostic service negative response or device-control request rejected."
    },
    {
        "pattern": ["cwod"],
        "error_bits": [14, 22],
        "root_cause": "CWOD calibration request rejected or calibration condition not met",
        "interpretation": "CWOD-specific request/calibration issue."
    },
    {
        "pattern": ["eeprom"],
        "error_bits": [17, 24],
        "root_cause": "EEPROM/classification readback mismatch",
        "interpretation": "Stored ECU values differ from expected rig/classification values."
    },
    {
        "pattern": ["torque"],
        "error_bits": [3, 11, 18],
        "root_cause": "Torque limit or torque verification mismatch",
        "interpretation": "Torque measurement, command, or verification behavior diverges."
    },
]


def identify_root_causes(divergence_df: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    if divergence_df.empty:
        return pd.DataFrame()

    high = divergence_df.sort_values("divergence_score", ascending=False).head(50)
    root_rows = []

    for _, r in high.iterrows():
        signal = str(r["signal"]).lower()
        matched = False

        for rule in ROOT_CAUSE_RULES:
            if all(p in signal for p in rule["pattern"]):
                for bit in rule["error_bits"]:
                    ec = ERROR_CODES.get(bit, {})
                    root_rows.append({
                        "log_name": r["log_name"],
                        "signal": r["signal"],
                        "divergence_score": r["divergence_score"],
                        "status": r["status"],
                        "probable_root_cause": rule["root_cause"],
                        "interpretation": rule["interpretation"],
                        "mapped_error_bit": bit,
                        "mapped_error_name": ec.get("name", ""),
                        "mapped_error_status_value": ec.get("status_value", ""),
                        "error_description": ec.get("description", ""),
                        "evidence": r["reason"],
                    })
                matched = True

        if not matched and r["divergence_score"] >= 60:
            root_rows.append({
                "log_name": r["log_name"],
                "signal": r["signal"],
                "divergence_score": r["divergence_score"],
                "status": r["status"],
                "probable_root_cause": "Unmapped high-divergence signal",
                "interpretation": "Signal diverges strongly from passed reference but no specific rule matched.",
                "mapped_error_bit": None,
                "mapped_error_name": "Review manually",
                "mapped_error_status_value": "",
                "error_description": "",
                "evidence": r["reason"],
            })

    root_df = pd.DataFrame(root_rows)
    if root_df.empty:
        return root_df

    root_df["rank_score"] = root_df["divergence_score"]
    root_df = root_df.sort_values("rank_score", ascending=False)
    return root_df.head(top_n)


# ============================================================
# Matrix construction
# ============================================================

BEHAVIOR_GROUPS = {
    "Ignition / RBS": ["ignition", "ign_txing", "rbs"],
    "Wheel speed auth": ["wheelspeeds", "angvelauth"],
    "Wheel speed invalid flags": ["wheelspeeds", "_inv"],
    "Wheel speeds / motion inputs": ["flspeed", "frspeed", "speed"],
    "GKN HIL motion panel": ["gkn_hil.panel"],
    "CAN / CAN-FD health": ["can"],
    "Software / config": ["version", "config", "mact"],
    "Diagnostics / service": ["diag", "diagnostic", "dtc", "negative", "request"],
    "Torque / clutch / calibration": ["torque", "clutch", "cwod", "cwo", "classification"],
    "EEPROM / NVM": ["eeprom", "nvm"],
}


def behavior_for_signal(signal: str) -> str:
    s = signal.lower()
    for behavior, keywords in BEHAVIOR_GROUPS.items():
        if all(k in s for k in keywords):
            return behavior

    # Looser fallback.
    if "wheel" in s:
        return "Wheel speeds / motion inputs"
    if "can" in s:
        return "CAN / CAN-FD health"
    if "ign" in s:
        return "Ignition / RBS"
    if "gkn_hil" in s:
        return "GKN HIL motion panel"
    return "Other signals"


def build_divergence_matrix(divergence_df: pd.DataFrame) -> pd.DataFrame:
    if divergence_df.empty:
        return pd.DataFrame()

    df = divergence_df.copy()
    df["behavior"] = df["signal"].apply(behavior_for_signal)

    agg = (
        df.groupby(["behavior", "log_name"])
        .agg(
            max_score=("divergence_score", "max"),
            top_signal=("signal", lambda x: x.iloc[0]),
            reasons=("reason", lambda x: "; ".join(pd.Series(x).dropna().astype(str).head(2)))
        )
        .reset_index()
    )

    def status_from_score(score):
        if score >= 60:
            return "❌"
        if score >= 25:
            return "⚠️"
        return "✅"

    agg["status"] = agg["max_score"].apply(status_from_score)
    agg["cell"] = agg.apply(
        lambda r: f"{r['status']} ({r['max_score']:.0f})",
        axis=1
    )

    matrix = agg.pivot(index="behavior", columns="log_name", values="cell").fillna("✅")
    matrix = matrix.reset_index().rename(columns={"behavior": "Signal / Behavior"})

    # Add interpretation column from highest divergence per behavior.
    interpretations = []
    for behavior in matrix["Signal / Behavior"]:
        sub = agg[agg["behavior"] == behavior].sort_values("max_score", ascending=False)
        if sub.empty:
            interpretations.append("Matches passed reference behavior")
        else:
            score = sub.iloc[0]["max_score"]
            reason = sub.iloc[0]["reasons"]
            if score >= 60:
                interpretations.append(f"Divergent from passed reference: {reason}")
            elif score >= 25:
                interpretations.append(f"Transitional / unstable: {reason}")
            else:
                interpretations.append("Matches passed reference behavior")

    matrix["Interpretation"] = interpretations
    return matrix


def heatmap_from_divergence(divergence_df: pd.DataFrame):
    if divergence_df.empty:
        return None

    top = (
        divergence_df.groupby("signal")["divergence_score"]
        .max()
        .sort_values(ascending=False)
        .head(40)
        .index
    )

    hdf = divergence_df[divergence_df["signal"].isin(top)].copy()
    pivot = hdf.pivot_table(
        index="signal",
        columns="log_name",
        values="divergence_score",
        aggfunc="max",
        fill_value=0
    )

    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale="RdYlGn_r",
        labels=dict(x="Log", y="Signal", color="Divergence"),
        title="Simultaneous Divergence Heatmap: Failed Logs vs Passed Reference"
    )
    fig.update_layout(height=max(500, 18 * len(pivot)))
    return fig


# ============================================================
# Optional Isolation Forest anomaly model across logs
# ============================================================

def isolation_forest_log_anomaly(all_logs: List[ParsedLog]) -> pd.DataFrame:
    """
    Log-level anomaly model using aggregate feature vectors.
    Works best with multiple passed and failed logs.
    """
    rows = []
    for log in all_logs:
        f = log.features
        if f.empty:
            continue
        rows.append({
            "log_name": log.name,
            "label": log.label,
            "n_signals": f["signal"].nunique(),
            "mean_zero_frac": pd.to_numeric(f["zero_frac"], errors="coerce").mean(),
            "mean_one_frac": pd.to_numeric(f["one_frac"], errors="coerce").mean(),
            "mean_transitions": pd.to_numeric(f["transitions"], errors="coerce").mean(),
            "mean_span": pd.to_numeric(f["span"], errors="coerce").mean(),
            "can_status_count": int(f["signal"].str.lower().str.contains("can").sum()),
            "inv_signal_count": int(f["signal"].str.lower().str.contains("_inv").sum()),
            "static_zero_signals": int(((pd.to_numeric(f["zero_frac"], errors="coerce") > 0.95) &
                                        (pd.to_numeric(f["span"], errors="coerce") == 0)).sum())
        })

    df = pd.DataFrame(rows)
    if len(df) < 3:
        df["iforest_score"] = np.nan
        df["iforest_flag"] = "Need >=3 logs"
        return df

    X_cols = [
        "n_signals", "mean_zero_frac", "mean_one_frac", "mean_transitions",
        "mean_span", "can_status_count", "inv_signal_count", "static_zero_signals"
    ]
    X = df[X_cols].fillna(0)

    contamination = min(0.45, max(0.05, len(df[df["label"] == "failed"]) / max(len(df), 1)))
    model = IsolationForest(random_state=42, contamination=contamination)
    pred = model.fit_predict(X)
    score = -model.score_samples(X)

    df["iforest_score"] = score
    df["iforest_flag"] = np.where(pred == -1, "Anomalous", "Normal")
    return df.sort_values("iforest_score", ascending=False)


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(
    page_title="AWD EOL Divergence Matrix & Root Cause Dashboard",
    layout="wide"
)

st.title("AWD EOL Anomaly Detection & Root Cause Dashboard")

st.markdown(
    """
Upload **passed reference logs** and **failed EOL logs**.  
The dashboard parses CANoe-style text logs, builds a passed reference profile,
detects failed-log divergence, maps likely AWD EOL error codes, and generates a
simultaneous divergence matrix.
"""
)

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
    high_threshold = st.slider("High divergence threshold", 40, 90, 60)
    warn_threshold = st.slider("Warning divergence threshold", 10, 50, 25)
    top_root_causes = st.slider("Root-cause rows to show", 3, 20, 8)

    st.header("3. Signal filter")
    signal_filter = st.text_input(
        "Only include signals containing",
        value="",
        help="Example: WheelSpeeds, GKN_HIL, IGN, CAN, torque"
    )


if not passed_files or not failed_files:
    st.info("Use the file browser in the left sidebar to upload at least one passed log and one failed log.")
    st.stop()


# Parse files.
passed_logs = []
failed_logs = []

with st.spinner("Parsing logs..."):
    for f in passed_files:
        text = read_uploaded_file(f)
        passed_logs.append(parse_log_text(text, f.name, "passed"))

    for f in failed_files:
        text = read_uploaded_file(f)
        failed_logs.append(parse_log_text(text, f.name, "failed"))

all_logs = passed_logs + failed_logs

# Show metadata.
st.subheader("Uploaded logs")

meta_rows = []
for log in all_logs:
    meta_rows.append({
        "log_name": log.name,
        "label": log.label,
        "events_parsed": len(log.events),
        "signals_parsed": 0 if log.features.empty else log.features["signal"].nunique(),
        "date": log.metadata.get("date", ""),
        "version": log.metadata.get("version", ""),
        "measurement_uuid": log.metadata.get("measurement_uuid", "")
    })
st.dataframe(pd.DataFrame(meta_rows), use_container_width=True)


# Build reference and compare.
ref_profile = build_reference_profile(passed_logs)

all_divergence = []
for log in failed_logs:
    d = compare_to_reference(log, ref_profile)
    if not d.empty:
        if signal_filter:
            d = d[d["signal"].str.contains(signal_filter, case=False, na=False)]
        all_divergence.append(d)

if not all_divergence:
    st.warning("No comparable signals were found. Check that passed and failed logs use similar signal naming.")
    st.stop()

divergence_df = pd.concat(all_divergence, ignore_index=True)

# Apply UI thresholds.
divergence_df["status"] = np.where(
    divergence_df["divergence_score"] >= high_threshold, "❌",
    np.where(divergence_df["divergence_score"] >= warn_threshold, "⚠️", "✅")
)
divergence_df["severity"] = np.where(
    divergence_df["divergence_score"] >= high_threshold, "High",
    np.where(divergence_df["divergence_score"] >= warn_threshold, "Medium", "Low")
)

# KPI row.
high_count = int((divergence_df["severity"] == "High").sum())
warn_count = int((divergence_df["severity"] == "Medium").sum())
ok_count = int((divergence_df["severity"] == "Low").sum())

k1, k2, k3, k4 = st.columns(4)
k1.metric("Failed logs", len(failed_logs))
k2.metric("High divergences", high_count)
k3.metric("Warnings", warn_count)
k4.metric("Matched / low", ok_count)


# Tabs.
tab_matrix, tab_root, tab_heatmap, tab_details, tab_model, tab_error = st.tabs([
    "Divergence Matrix",
    "Root Cause",
    "Heatmap",
    "Signal Details",
    "Log-Level Model",
    "Error Code Reference"
])

with tab_matrix:
    st.subheader("Simultaneous Divergence Matrix")

    st.markdown(
        """
Legend: ✅ matches passed reference behavior &nbsp;&nbsp; ❌ divergent from passed reference &nbsp;&nbsp; ⚠️ transitional / unstable.
        """
    )

    matrix = build_divergence_matrix(divergence_df)
    st.dataframe(matrix, use_container_width=True, hide_index=True)

    csv = matrix.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download divergence matrix CSV",
        data=csv,
        file_name="awd_eol_divergence_matrix.csv",
        mime="text/csv"
    )

with tab_root:
    st.subheader("Probable Root Cause Identification")

    root_df = identify_root_causes(divergence_df, top_n=top_root_causes)

    if root_df.empty:
        st.success("No high-confidence root cause identified from the current thresholds.")
    else:
        st.dataframe(root_df.drop(columns=["rank_score"], errors="ignore"), use_container_width=True)

        top = root_df.iloc[0]
        st.markdown("### Top hypothesis")
        st.error(
            f"""
**Root cause:** {top['probable_root_cause']}  
**Evidence signal:** `{top['signal']}`  
**Mapped EOL error:** Bit {top['mapped_error_bit']} — {top['mapped_error_name']}  
**Evidence:** {top['evidence']}
"""
        )

        csv = root_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download root-cause report CSV",
            data=csv,
            file_name="awd_eol_root_cause_report.csv",
            mime="text/csv"
        )

with tab_heatmap:
    st.subheader("Signal Divergence Heatmap")

    fig = heatmap_from_divergence(divergence_df)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No heatmap available.")

with tab_details:
    st.subheader("Signal-Level Divergence Details")

    sort_col = st.selectbox(
        "Sort by",
        ["divergence_score", "signal", "log_name", "severity"],
        index=0
    )

    st.dataframe(
        divergence_df.sort_values(sort_col, ascending=False),
        use_container_width=True,
        hide_index=True
    )

    csv = divergence_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download signal divergence detail CSV",
        data=csv,
        file_name="awd_eol_signal_divergence_detail.csv",
        mime="text/csv"
    )

with tab_model:
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

with tab_error:
    st.subheader("AWD EOL Error Code Reference")

    error_df = pd.DataFrame([
        {
            "Error Bit": bit,
            "Error Status Value": info["status_value"],
            "Error Name": info["name"],
            "Description": info["description"],
            "Root Cause Hint": info["root_hint"],
        }
        for bit, info in ERROR_CODES.items()
    ])

    st.dataframe(error_df, use_container_width=True, hide_index=True)