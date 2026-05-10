# Data models and type definitions for the Anomaly Detection Dashboard

from dataclasses import dataclass
from typing import Dict, List, Set, Optional, Union, Tuple
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import FeatureHasher


@dataclass
class PassModelBundle:
    """Container for trained anomaly detection model components."""
    scaler: StandardScaler
    model: IsolationForest
    hasher: FeatureHasher
    score_p05: float
    score_p95: float
    pass_signal_set: Set[str]


@dataclass
class ParsedLog:
    """Container for parsed log data and metadata."""
    name: str
    label: str
    raw_text: str
    events: pd.DataFrame
    features: pd.DataFrame
    metadata: Dict[str, str]


@dataclass
class DivergenceResult:
    """Container for signal divergence analysis results."""
    signal: str
    log_name: str
    divergence_score: float
    status: str
    severity: str
    evidence: List[str]
    feature_values: Dict[str, float]


@dataclass
class RootCauseHypothesis:
    """Container for root cause analysis results."""
    probable_root_cause: str
    signal: str
    mapped_error_bit: int
    mapped_error_name: str
    evidence: str
    rank_score: float