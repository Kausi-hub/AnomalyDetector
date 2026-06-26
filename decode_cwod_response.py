def decode_cwod_response(data_bytes):
    """
    data_bytes = list of ints (e.g. [0x71,0x03,0x04,0x5F,0x01,0x00,0xA8])
    """

    result = {}

    # Basic fields
    result['service'] = hex(data_bytes[0])
    result['sub_function'] = hex(data_bytes[1])
    result['routine_id'] = hex(data_bytes[2] << 8 | data_bytes[3])
    result['status'] = data_bytes[4]

    # Status decode
    status_map = {
        0x00: "Not started",
        0x01: "Completed",
        0x02: "Running",
        0x03: "Aborted"
    }
    result['status_meaning'] = status_map.get(data_bytes[4], "Unknown")

    # Result bytes
    fault_word = (data_bytes[5] << 8) | data_bytes[6]
    result['fault_word'] = hex(fault_word)

    # Bit-level decode
    fault_flags = {
        7: "Calibration failure",
        6: "Timeout",
        5: "Precondition failed",
        4: "CWO out of range",
        3: "Signal invalid",
        2: "Hardware fault",
        1: "Communication/authentication fault",
        0: "Unknown/vendor specific"
    }

    active_faults = []
    for bit, desc in fault_flags.items():
        if fault_word & (1 << bit):
            active_faults.append(desc)

    result['active_faults'] = active_faults

    return result


# Example usage
data = [0x71,0x03,0x04,0x5F,0x01,0x00,0xA8]
print(decode_cwod_response(data))