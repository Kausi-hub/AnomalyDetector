import re
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox


# ---------------------------
# CWOD Decode Logic
# ---------------------------
def decode_cwod_response(data_bytes):
    result = {}

    result['service'] = hex(data_bytes[0])
    result['sub_function'] = hex(data_bytes[1])
    result['routine_id'] = hex((data_bytes[2] << 8) | data_bytes[3])
    result['status'] = data_bytes[4]

    status_map = {
        0x00: "Not started",
        0x01: "Completed",
        0x02: "Running",
        0x03: "Aborted"
    }
    result['status_meaning'] = status_map.get(data_bytes[4], "Unknown")

    fault_word = (data_bytes[5] << 8) | data_bytes[6]
    result['fault_word_hex'] = hex(fault_word)
    result['fault_word_dec'] = fault_word

    # Bit decoding
    fault_flags = {
        7: "Calibration failure",
        6: "Timeout",
        5: "Precondition failed",
        4: "CWO out of range",
        3: "Signal invalid",
        2: "Hardware fault",
        1: "Comm/auth fault",
        0: "Unknown"
    }

    active_faults = []
    for bit, desc in fault_flags.items():
        if (fault_word >> bit) & 1:
            active_faults.append(desc)

    result['active_faults'] = ", ".join(active_faults) if active_faults else "None"
    result['verdict'] = "PASS" if fault_word == 0 else "FAIL"

    return result


# ---------------------------
# Log File Parsing
# ---------------------------
def extract_and_decode(log_file_path):
    results = []

    with open(log_file_path, 'r', errors='ignore') as file:
        for line in file:

            line_lower = line.lower()

            # Filter only Rx + RDCM response
            if "rx" not in line_lower or "14daf11a" not in line_lower:
                continue

            parts = line.split()

            try:
                # Extract timestamp (first column)
                timestamp = float(parts[0])

                # Find payload "8 XX XX XX ..."
                raw_bytes = None
                for i in range(len(parts) - 9):
                    if parts[i] == "8":
                        candidate = parts[i+1:i+9]

                        # Check valid hex format
                        if all(len(x) == 2 for x in candidate):
                            raw_bytes = candidate
                            break

                if raw_bytes is None:
                    continue

                # Convert to integer bytes
                data_bytes = [int(b, 16) for b in raw_bytes]

                # Remove PCI (first byte = length)
                if data_bytes[0] <= 0x08:
                    data_bytes = data_bytes[1:]

                if len(data_bytes) < 7:
                    continue

                service_id = data_bytes[0]

                # Only 31 / 71 service request
                if service_id not in [0x31, 0x71]:
                    continue

                decoded = {}
                decoded['timestamp'] = timestamp
                decoded['raw'] = " ".join(raw_bytes)
                decoded['service'] = hex(service_id)
                decoded['type'] = "Response" if service_id == 0x71 else "Request"

                # ------------------------------------
                # RESPONSE (0x71)
                # ------------------------------------
                if service_id == 0x71:

                    decoded['sub_function'] = hex(data_bytes[1])
                    decoded['routine_id'] = hex((data_bytes[2] << 8) | data_bytes[3])

                    # Filter CWOD only
                    if decoded['routine_id'] != '0x45f':
                        continue

                    status = data_bytes[4]
                    decoded['status'] = status

                    status_map = {
                        0x00: "Not started",
                        0x01: "Completed",
                        0x02: "Running",
                        0x03: "Aborted"
                    }
                    decoded['status_meaning'] = status_map.get(status, "Unknown")

                    # Fault word
                    fault_word = (data_bytes[5] << 8) | data_bytes[6]
                    decoded['fault_word_hex'] = hex(fault_word)
                    decoded['fault_word_dec'] = fault_word

                    # Bit decode
                    fault_flags = {
                        7: "Calibration failure",
                        6: "Timeout",
                        5: "Precondition failed",
                        4: "CWO out of range",
                        3: "Signal invalid",
                        2: "Hardware fault",
                        1: "Comm/auth fault",
                        0: "Unknown"
                    }

                    active_faults = []
                    for bit, desc in fault_flags.items():
                        if (fault_word >> bit) & 1:
                            active_faults.append(desc)

                    decoded['active_faults'] = ", ".join(active_faults) if active_faults else "None"
                    decoded['verdict'] = "PASS" if fault_word == 0 else "FAIL"

                # ------------------------------------
                # REQUEST (0x31)
                # ------------------------------------
                elif service_id == 0x31:

                    decoded['sub_function'] = hex(data_bytes[1])
                    decoded['routine_id'] = hex((data_bytes[2] << 8) | data_bytes[3])

                    if decoded['routine_id'] != '0x45f':
                        continue

                    decoded['status'] = "Request"

                results.append(decoded)

            except:
                continue

    return pd.DataFrame(results)


# ---------------------------
# GUI Functions
# ---------------------------
def select_file():
    file_path = filedialog.askopenfilename(
        title="Select CAN Log File",
        filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
    )

    if not file_path:
        return

    try:
        df = extract_and_decode(file_path)

        if df.empty:
            messagebox.showwarning("No Data", "No CWOD responses found in file.")
            return

        # Save output
        output_file = "CWOD_Decode_Output.xlsx"
        df.to_excel(output_file, index=False)

        # Show summary
        summary = f"""
        File Processed Successfully!

        Total CWOD Responses: {len(df)}
        FAIL Count: {len(df[df['verdict'] == 'FAIL'])}
        PASS Count: {len(df[df['verdict'] == 'PASS'])}

        Output saved as:
        {output_file}
        """

        messagebox.showinfo("Success", summary)

    except Exception as e:
        messagebox.showerror("Error", str(e))


# ---------------------------
# GUI Layout
# ---------------------------
root = tk.Tk()
root.title("CWOD Decoder Tool")
root.geometry("400x200")

label = tk.Label(
    root,
    text="CWOD Diagnostic Decoder\nSelect CAN Log File",
    font=("Arial", 12)
)
label.pack(pady=20)

btn = tk.Button(
    root,
    text="Browse Log File",
    command=select_file,
    font=("Arial", 10),
    width=20,
    height=2
)
btn.pack()

root.mainloop()