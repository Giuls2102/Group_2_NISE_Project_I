from datetime import datetime
import serial
import serial.tools.list_ports
import time
import sys
import threading
from pynput import keyboard
from pathlib import Path
import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================
# String V1/V2 board uses to identify itself on USB:
USB_DEVICE_DESCRIPTION = "USB-Enhanced-SERIAL CH9102"
# USB_DEVICE_DESCRIPTION = "Silicon Labs CP210x"  # (For V1)
# USB_DEVICE_DESCRIPTION = "USB-Enhanced-Serial CH9102"  # (For V2)

BAUD_RATE = 115200  # ESP32 board's serial baud rate
TIME_PUSH_PERIOD_SEC = 30   # Send local time to receiver ESP32 board every 30 s to keep RTC updated
CSV_HEADER = (  # must match the structure sent by the receiver ESP32 board
    "ts_epoch_ms,ts_uptime_ms,"
    "imu1_acc_x,imu1_acc_y,imu1_acc_z,"
    "imu1_gyr_x,imu1_gyr_y,imu1_gyr_z,"
    "imu1_pitch,imu1_roll,"
    "imu2_acc_x,imu2_acc_y,imu2_acc_z,"
    "imu2_gyr_x,imu2_gyr_y,imu2_gyr_z,"
    "imu2_pitch,imu2_roll"
)
# =============================================================================

# ========== FIND SERIAL PORT ==========
def find_esp32_port():
    """
    Automatically finds the COM port for the ESP32 board.
    """
    print(f"Searching for COM port with description: '{USB_DEVICE_DESCRIPTION}'...")
    ports = serial.tools.list_ports.comports()
    for port, desc, hwid in sorted(ports):
        if USB_DEVICE_DESCRIPTION in desc:
            print(f"Found! Port: {port}, Description: {desc}")
            return port
    print("--- ERROR: Board not found. ---")
    print("Is the board plugged in? Is the description correct?")
    return None

# ========== GLOBAL STATE ==========
PORT_NAME = None  # to store the found serial port name 
ser = None  # to hold the serial.Serial instance
stop_event = threading.Event()  # to signal threads to stop

# keyboard state
is_recording = False  # whether we are currently recording samples
log_rows = []   # list of strings without the leading "CSV,"
n_esc_presses = 0  # count of how many times ESC was pressed (trial number) for CSV file naming

# ========== UTILITIES ==========
# Get local epoch in ms (for CET timezone or system local timezone)
def now_local_epoch_ms() -> int:
    """Return local (CET/your system zone) epoch in ms."""
    epoch_ms_utc = int(time.time() * 1000)
    offset_ms = int(datetime.now().astimezone().utcoffset().total_seconds() * 1000)
    return epoch_ms_utc + offset_ms

# Send time frame to receiver
def send_time_frame():
    """Send a single T<epoch_ms_local> frame to the receiver."""
    if not ser:
        return
    frame = f"T{now_local_epoch_ms()}\n".encode("ascii")
    ser.write(frame)
    ser.flush()

# Safe Pearson correlation computation
def _safe_corr(xs, ys):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.size < 2 or ys.size < 2:
        return float('nan')
    # guard zero-variance
    if np.allclose(xs, xs[0]) or np.allclose(ys, ys[0]):
        return float('nan')
    return float(np.corrcoef(xs, ys)[0, 1])

# Compute correlations from log rows
def _compute_corrs_from_log_rows(rows):
    """
    Compute imu1<->imu2 Pearson correlations for:
    acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, pitch, roll
    using the payloads in log_rows (strings without 'CSV,').
    """
    # Expected 18 columns based on CSV_HEADER
    N_COLS = 18
    cols = [[] for _ in range(N_COLS)]
    for line in rows:
        parts = line.split(',')
        if len(parts) != N_COLS:
            continue
        for i, p in enumerate(parts):
            try:
                cols[i].append(float(p))
            except Exception:
                pass
    # Define column index pairs for correlation
    pairs = {
        "acc_x": (2, 10),
        "acc_y": (3, 11),
        "acc_z": (4, 12),
        "gyr_x": (5, 13),
        "gyr_y": (6, 14),
        "gyr_z": (7, 15),
        "pitch": (8, 16),
        "roll": (9, 17),
    }
    return {k: _safe_corr(cols[i], cols[j]) for k, (i, j) in pairs.items()}

# ========== THREADS ==========
def serial_reader():
    """Read lines from Serial; log those starting with 'CSV,' when recording."""
    global log_rows
    buf = b""
    while not stop_event.is_set():
        try:
            line = ser.readline()  # reads until '\n' or timeout
            if not line:
                continue
            try:
                s = line.decode("ascii", "ignore").strip()
            except Exception:
                continue

            # Only capture the compact CSV line we added to the receiver:
            if s.startswith("CSV,"):
                payload = s[4:]  # drop "CSV,"
                if is_recording:
                    log_rows.append(payload)
            # Optional: Print status heartbeat
            # if s.startswith("CSV,"):
            #     print(f"[sample] {s[:64]}...")

        except serial.SerialException as e:
            print(f"[Serial read error] {e}")
            break
        except Exception:
            pass

def time_pusher():
    """Send local epoch to the receiver now and every TIME_PUSH_PERIOD_SEC."""
    # initial push
    send_time_frame()
    while not stop_event.is_set():
        for _ in range(TIME_PUSH_PERIOD_SEC * 10):
            if stop_event.is_set():
                return
            time.sleep(0.1)
        send_time_frame()

# ========== KEYBOARD LISTENER ==========
def on_press(key):
    global is_recording, log_rows
    try:
        if hasattr(key, "char") and key.char:
            ch = key.char
            # --- handle 's' / 'S' (start recording) ---
            if ch in ('s', 'S') and not is_recording:
                is_recording = True
                log_rows = []  # reset buffer
                print("Recording: STARTED (press ESC to save).")

    except AttributeError:
        # special keys without .char
        pass

def on_release(key):
    global is_recording, log_rows, n_esc_presses
    try:
        # ESC pressed: save CSV and stop recording
        if key == keyboard.Key.esc:
            if is_recording:
                is_recording = False
                n_esc_presses += 1
                # Save to CSV
                ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                out = Path(f"IMU_trial_{n_esc_presses}_date_{ts}.csv").resolve()
                # Compute correlations once for the whole recording
                corrs = _compute_corrs_from_log_rows(log_rows)
                _corr_cols = ["acc_x_corr","acc_y_corr","acc_z_corr","gyr_x_corr",
                              "gyr_y_corr","gyr_z_corr","pitch_corr","roll_corr"]
                # Augment header with left/right labels and correlation columns
                CSV_HEADER_LR = (  # Left/Right IMU labels
                    "ts_epoch_ms,ts_uptime_ms,"
                    "imuR_acc_x,imuR_acc_y,imuR_acc_z,"
                    "imuR_gyr_x,imuR_gyr_y,imuR_gyr_z,"
                    "imuR_pitch,imuR_roll,"
                    "imuL_acc_x,imuL_acc_y,imuL_acc_z,"
                    "imuL_gyr_x,imuL_gyr_y,imuL_gyr_z,"
                    "imuL_pitch,imuL_roll"
                )
                # Prepare augmented header string
                _aug_header = CSV_HEADER_LR + "," + ",".join(_corr_cols)
                _corr_values_ordered = [
                    corrs.get("acc_x", float('nan')),
                    corrs.get("acc_y", float('nan')),
                    corrs.get("acc_z", float('nan')),
                    corrs.get("gyr_x", float('nan')),
                    corrs.get("gyr_y", float('nan')),
                    corrs.get("gyr_z", float('nan')),
                    corrs.get("pitch", float('nan')),
                    corrs.get("roll", float('nan')),
                ]
                # Prepare correlation suffix string
                _corr_suffix = "," + ",".join(
                    (f"{v:.6f}" if np.isfinite(v) else "nan") for v in _corr_values_ordered
                )
                # Write CSV and save
                with open(out, "w", encoding="utf-8") as f:
                    f.write(_aug_header + "\n")  # Augmented header with corr columns
                    for row in log_rows:
                        # Append the same per-dataset correlation values to every row
                        f.write(row + _corr_suffix + "\n")
                print(f"Recording: SAVED {len(log_rows)} rows to {out}")
                log_rows = []
            else:
                print("ESC pressed: not recording; nothing to save.")

    except Exception as e:
        print(f"[Key release error] {e}")

# ========== MAIN ==========
def main():
    global PORT_NAME, ser
    PORT_NAME = find_esp32_port()
    if not PORT_NAME:
        time.sleep(3)
        sys.exit(1)

    # Open the serial port ONCE and keep it open for logging + commands
    try:
        ser = serial.Serial(PORT_NAME, BAUD_RATE, timeout=0.2)
        try:
            # Avoid toggling reset lines on some ESP32 boards
            ser.setDTR(False); ser.setRTS(False)
        except Exception:
            pass
        time.sleep(0.05)  # wait for stable connection
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        print(f"Using {PORT_NAME}.")
        print("Press 's' to START recording)")
        print("Press ESC to SAVE to CSV.")
        print("(Close the Arduino Serial Monitor while this logger is running.)\n")
        print("Press Ctrl+C to exit.\n")

        # Start threads
        t_reader = threading.Thread(target=serial_reader, daemon=True)
        t_pusher = threading.Thread(target=time_pusher, daemon=True)
        t_reader.start()
        t_pusher.start()

        # Keyboard listener (non-blocking)
        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()

        # Keep main alive
        while True:
            time.sleep(1)  # To prevent busy-waiting and reduce CPU usage.

    except serial.SerialException as e:
        print(f"\n--- SERIAL ERROR ---")
        print(f"Failed to use port {PORT_NAME}. Is it in use?")
        print(f"Close Arduino Serial Monitor and retry.")
        print(f"Details: {e}")
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        stop_event.set()
        try:
            if ser and ser.is_open:
                ser.close()
        except Exception:
            pass
        print("Goodbye.")

if __name__ == "__main__":
    main()
