# Configuration constants and data structures for the Anomaly Detection Dashboard

# Encoding options for file reading
ENCODINGS = ["utf-8", "latin-1", "cp1252"]

# ML Model defaults
DEFAULT_CONTAMINATION = 0.03
DEFAULT_N_ESTIMATORS = 200
RANDOM_STATE = 42
HASH_FEATURES = 64

# UI defaults
HIGH_THRESHOLD_DEFAULT = 60
WARN_THRESHOLD_DEFAULT = 25
TOP_ROOT_CAUSES_DEFAULT = 8

# Feature columns for analysis
FEATURE_COLS = [
    "count", "numeric_count", "last", "mean", "std", "min", "max", "span",
    "zero_frac", "one_frac", "nonzero_frac", "unique_count",
    "transitions", "duration", "event_rate_hz", "first_seen", "last_seen"
]

# Behavior groups for signal categorization
BEHAVIOR_GROUPS = {
    "Ignition / RBS": ["ignition", "ign_txing", "rbs"],
    "Wheel speed auth": ["wheelspeeds", "angvelauth"],
    "Wheel speed invalid flags": ["wheelspeeds", "_inv"],
    "Wheel speeds / motion inputs": ["flspeed", "frspeed", "speed"],
    "HIL motion panel": ["hil.panel"],
    "CAN / CAN-FD health": ["can"],
    "Software / config": ["version", "config", "mact"],
    "Diagnostics / service": ["diag", "diagnostic", "dtc", "negative", "request"],
    "Torque / clutch / calibration": ["torque", "clutch", "cwod", "cwo", "classification"],
    "EEPROM / NVM": ["eeprom", "nvm"],
}

# Error codes mapping
ERROR_CODES = {
    0: {
        "status_value": "2^0",
        "name": "No Error",
        "description": "No error condition detected.",
        "root_hint": "Normal operation."
    },
    1: {
        "status_value": "2^1",
        "name": "CAN Communication Error",
        "description": "CAN communication lost or corrupted.",
        "root_hint": "CAN bus / network issue."
    },
    2: {
        "status_value": "2^2",
        "name": "Torque Sensor Error",
        "description": "Torque sensor reading out of range or invalid.",
        "root_hint": "Torque sensor hardware / calibration."
    },
    3: {
        "status_value": "2^3",
        "name": "Speed Sensor Error",
        "description": "Wheel speed sensor reading invalid.",
        "root_hint": "Wheel speed sensor hardware."
    },
    4: {
        "status_value": "2^4",
        "name": "Thermal Protection",
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
        "name": "Classification Values Don't match between Rig and ECU",
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
        "name": "CAN-FD Communication Error",
        "description": "CAN-FD communication lost or corrupted.",
        "root_hint": "CAN-FD bus / network issue."
    },
    22: {
        "status_value": "2^22",
        "name": "HIL Panel Communication Error",
        "description": "HIL motion panel communication lost.",
        "root_hint": "HIL panel / network issue."
    },
    23: {
        "status_value": "2^23",
        "name": "EEPROM Write Error",
        "description": "Failed to write calibration data to EEPROM.",
        "root_hint": "NVM / EEPROM write failure."
    },
    24: {
        "status_value": "2^24",
        "name": "Software Version Mismatch",
        "description": "Software version does not match expected version.",
        "root_hint": "Software version / compatibility issue."
    },
    25: {
        "status_value": "2^25",
        "name": "Configuration Error",
        "description": "Configuration parameters out of range or invalid.",
        "root_hint": "Configuration / parameter issue."
    },
    26: {
        "status_value": "2^26",
        "name": "Timeout Error",
        "description": "Operation timed out before completion.",
        "root_hint": "Timeout / timing issue."
    },
    27: {
        "status_value": "2^27",
        "name": "Invalid Data Error",
        "description": "Received invalid or corrupted data.",
        "root_hint": "Data integrity / corruption issue."
    }
}

# Root cause rules mapping
ROOT_CAUSE_RULES = {
    "CAN Communication Error": {
        "signals": ["can", "canfd", "communication"],
        "error_bits": [1, 13, 21],
        "boost_factor": 2.0
    },
    "Torque Sensor Error": {
        "signals": ["torque", "sensor", "adc"],
        "error_bits": [2],
        "boost_factor": 2.0
    },
    "Speed Sensor Error": {
        "signals": ["speed", "wheel", "sensor", "angvel"],
        "error_bits": [3],
        "boost_factor": 2.0
    },
    "Thermal Protection": {
        "signals": ["temp", "thermal", "clutch_temp"],
        "error_bits": [4],
        "boost_factor": 2.0
    },
    "Temperature Precondition": {
        "signals": ["ambient", "temp", "initial"],
        "error_bits": [5],
        "boost_factor": 1.5
    },
    "Mechanical Docking": {
        "signals": ["dock", "shaft", "mechanical"],
        "error_bits": [6],
        "boost_factor": 2.0
    },
    "Run-in Energy": {
        "signals": ["runin", "energy", "motion", "torque"],
        "error_bits": [7],
        "boost_factor": 1.8
    },
    "Clutch Wear-off": {
        "signals": ["cwo", "cwod", "clutch", "wear"],
        "error_bits": [8, 9],
        "boost_factor": 2.0
    },
    "Classification Values": {
        "signals": ["classification", "learn", "calibration"],
        "error_bits": [10, 17],
        "boost_factor": 1.8
    },
    "Torque Verification": {
        "signals": ["torque", "verification", "tolerance"],
        "error_bits": [11],
        "boost_factor": 2.0
    },
    "Power Supply": {
        "signals": ["voltage", "power", "supply"],
        "error_bits": [12],
        "boost_factor": 1.5
    },
    "Network Interruption": {
        "signals": ["interrupt", "network", "disturbance"],
        "error_bits": [13],
        "boost_factor": 1.8
    },
    "Diagnostic Rejection": {
        "signals": ["diag", "diagnostic", "request", "negative"],
        "error_bits": [14],
        "boost_factor": 1.5
    },
    "NVM Write Rejection": {
        "signals": ["nvm", "eeprom", "write"],
        "error_bits": [15],
        "boost_factor": 1.8
    },
    "ECU Initialization": {
        "signals": ["ecu", "etm", "operation", "state"],
        "error_bits": [16],
        "boost_factor": 2.0
    },
    "Open Loop Torque": {
        "signals": ["openloop", "torque", "command"],
        "error_bits": [18],
        "boost_factor": 1.8
    },
    "Dynamic Response": {
        "signals": ["step", "response", "dynamic"],
        "error_bits": [19],
        "boost_factor": 1.8
    },
    "Rig Motor Speed": {
        "signals": ["rig", "motor", "speed", "command"],
        "error_bits": [20],
        "boost_factor": 1.8
    },
    "HIL Communication": {
        "signals": ["hil", "panel", "motion"],
        "error_bits": [22],
        "boost_factor": 1.5
    },
    "EEPROM Write": {
        "signals": ["eeprom", "nvm", "write", "calibration"],
        "error_bits": [23],
        "boost_factor": 1.8
    },
    "Software Version": {
        "signals": ["version", "software", "compatibility"],
        "error_bits": [24],
        "boost_factor": 1.5
    },
    "Configuration": {
        "signals": ["config", "parameter", "configuration"],
        "error_bits": [25],
        "boost_factor": 1.5
    },
    "Timeout": {
        "signals": ["timeout", "timing"],
        "error_bits": [26],
        "boost_factor": 1.2
    },
    "Data Corruption": {
        "signals": ["data", "corrupt", "invalid"],
        "error_bits": [27],
        "boost_factor": 1.2
    }
}