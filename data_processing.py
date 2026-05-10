# Data processing functions for log parsing and feature extraction

import re
import math
from typing import Dict, List, Optional, Union, Tuple, Set
import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import StandardScaler

from config import ENCODINGS, FEATURE_COLS, HASH_FEATURES, DEFAULT_CONTAMINATION, DEFAULT_N_ESTIMATORS, RANDOM_STATE
from data_models import ParsedLog


def read_uploaded_file(uploaded_file) -> str:
    """Read uploaded file content with multiple encoding fallbacks."""
    data = uploaded_file.read()
    for encoding in ENCODINGS:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def normalize_signal_name(name: str) -> str:
    """Normalize signal names for consistent processing."""
    name = name.strip()
    name = name.replace("\\_", "_")
    name = name.replace("::", ".")
    name = re.sub(r"\s+", "_", name)
    return name


def parse_value(value_str: str) -> Union[float, str, np.float64]:
    """Parse string values into appropriate numeric or string types."""
    value_str = value_str.strip().strip('"')
    if value_str == "":
        return np.nan

    # Keep arrays and hex payloads as strings
    if "[" in value_str or "]" in value_str:
        return value_str

    # Attempt numeric conversion
    try:
        if re.match(r"^-?\d+(\.\d+)?$", value_str):
            return float(value_str)
    except Exception:
        pass

    return value_str


def extract_metadata(text: str) -> Dict[str, str]:
    """Extract metadata from log file comments."""
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

    return metadata


def parse_sv_signal(payload: str) -> Optional[Dict]:
    """Parse SV (System Variable) style signals."""
    sv_pattern = re.compile(r"^SV:\s+.*?(::[A-Za-z0-9_:]+)\s*=\s*(.*)$")
    sv_match = sv_pattern.match(payload)
    if sv_match:
        signal = normalize_signal_name(sv_match.group(1).lstrip(":"))
        value = parse_value(sv_match.group(2))
        return {
            "signal": signal,
            "value": value,
            "event_type": "SV"
        }
    return None


def parse_assign_signal(payload: str) -> Optional[Dict]:
    """Parse assignment-style signals (signal := value)."""
    assign_pattern = re.compile(r"^([A-Za-z0-9_:\\.\-\[\]/]+)\s*:=\s*(.*)$")
    assign_match = assign_pattern.match(payload)
    if assign_match:
        signal = normalize_signal_name(assign_match.group(1))
        value = parse_value(assign_match.group(2))
        return {
            "signal": signal,
            "value": value,
            "event_type": "ASSIGN"
        }
    return None


def parse_can_status(payload: str) -> Optional[Dict]:
    """Parse CAN status messages."""
    can_status_pattern = re.compile(r"^(CAN(?:FD)?\s+\d+)\s+Status:(.*)$", re.IGNORECASE)
    can_match = can_status_pattern.match(payload)
    if can_match:
        signal = normalize_signal_name(can_match.group(1) + "_Status")
        value = can_match.group(2).strip()
        return {
            "signal": signal,
            "value": value,
            "event_type": "CAN_STATUS"
        }
    return None


def parse_can_frame(payload: str) -> Optional[Dict]:
    """Parse CAN frame transmissions."""
    can_tx_pattern = re.compile(r"^(CANFD|CAN)\s+(\d+)\s+(Tx|Rx)\s+([0-9A-Fa-fx]+)\s+([A-Za-z0-9_]+)?\s*(.*)$")
    can_match = can_tx_pattern.match(payload)
    if can_match:
        bus_type, channel, direction, can_id, pdu, rest = can_match.groups()
        signal = normalize_signal_name(f"{bus_type}{channel}_{direction}_{pdu or can_id}")
        return {
            "signal": signal,
            "value": 1.0,
            "event_type": "CAN_FRAME"
        }
    return None


def parse_log_line(line: str) -> Optional[Dict]:
    """Parse a single log line and extract signal data."""
    line_pattern = re.compile(r"^\s*(\d+\.\d+)\s+(.*)$")
    match = line_pattern.match(line)
    if not match:
        return None

    timestamp = float(match.group(1))
    payload = match.group(2).strip()

    # Try different parsing strategies
    parsers = [parse_sv_signal, parse_assign_signal, parse_can_status, parse_can_frame]
    for parser in parsers:
        result = parser(payload)
        if result:
            result["time"] = timestamp
            result["raw"] = line
            return result

    return None


def parse_log_text(text: str, name: str, label: str) -> ParsedLog:
    """
    Parse CANoe-style text logs into structured data.

    Supports multiple log formats:
    - Signal assignments: signal := value
    - SV variables: SV: ... ::Signal = value
    - CAN status: CAN 1 Status: message
    - CAN frames: CANFD 1 Tx/Rx ...
    """
    events_data = []
    metadata = extract_metadata(text)

    for line in text.splitlines():
        parsed_line = parse_log_line(line)
        if parsed_line:
            events_data.append(parsed_line)

    events = pd.DataFrame(events_data)

    if events.empty:
        features = pd.DataFrame()
    else:
        events["log_name"] = name
        events["label"] = label
        features = compute_features(events)

    return ParsedLog(name=name, label=label, raw_text=text, events=events, features=features, metadata=metadata)


def to_numeric_series(series: pd.Series) -> pd.Series:
    """Convert series to numeric, handling errors gracefully."""
    return pd.to_numeric(series, errors="coerce")


def compute_features(events: pd.DataFrame) -> pd.DataFrame:
    """Compute statistical features for each signal from event data."""
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


def _build_training_matrix(passed_logs: List[ParsedLog],
                           feature_cols: List[str] = FEATURE_COLS,
                           n_hash_features: int = HASH_FEATURES) -> Tuple[pd.DataFrame, np.ndarray, FeatureHasher, Set[str]]:
    """Build training matrix from passed logs for ML model training."""
    rows = []
    for log in passed_logs:
        if log.features.empty:
            continue
        f = log.features.copy()
        f["log_name"] = log.name
        rows.append(f)

    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if df.empty:
        return df, np.empty((0, 0)), FeatureHasher(n_features=n_hash_features, input_type="string"), set()

    # Numeric feature matrix
    X_num = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()

    # Hash the signal name into a fixed numeric vector
    hasher = FeatureHasher(n_features=n_hash_features, input_type="string", alternate_sign=False)
    signals = df["signal"].astype(str).tolist()
    X_sig = hasher.transform([[s] for s in signals]).toarray()

    # Combine
    X = np.hstack([X_num, X_sig])

    pass_signal_set = set(df["signal"].astype(str).unique())
    return df, X, hasher, pass_signal_set