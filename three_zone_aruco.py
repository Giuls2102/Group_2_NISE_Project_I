#!/usr/bin/env python3
import sys, time, threading, subprocess, shlex
from collections import deque
from datetime import datetime
import numpy as np
import cv2

# === Unity JSON export helpers ===============================================
import json, os, pathlib, tempfile, time as _time

# top of the file, where OUT_DIR / TIMES_ALIAS are defined
OUT_DIR = "/Users/giulia/Desktop/MSNE/MSNE/Term_3/NISE/ClassroomIntro"
TIMES_ALIAS = "times_trial.json"
OUT_PATH = os.path.join(OUT_DIR, TIMES_ALIAS)
print(f"[times] saving to: {OUT_PATH}", flush=True)

_trial_t0 = _time.perf_counter()  # trial-relative zero
_state = {
    # per-book completion times (seconds since trial start)
    "Book1": None,
    "Book2": None,
    "Book3": None,
    # attempts/overall
    "overall_attempts_total": 0,
    "overall_time_s": 0.0,
    # NEW: per-book start/end times (seconds since trial start)
    "start_Book1": None, "end_Book1": None,
    "start_Book2": None, "end_Book2": None,
    "start_Book3": None, "end_Book3": None,
}

def _rel():  # monotonic, relative to trial start
    return _time.perf_counter() - _trial_t0

def _write_times():
    pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    payload = {
        "Book1": -1.0 if _state["Book1"] is None else float(_state["Book1"]),
        "Book2": -1.0 if _state["Book2"] is None else float(_state["Book2"]),
        "Book3": -1.0 if _state["Book3"] is None else float(_state["Book3"]),
        "overall_attempts_total": int(_state["overall_attempts_total"]),
        "overall_time_s": float(_state["overall_time_s"]),
        "start_Book1": -1.0 if _state["start_Book1"] is None else float(_state["start_Book1"]),
        "end_Book1":   -1.0 if _state["end_Book1"]   is None else float(_state["end_Book1"]),
        "start_Book2": -1.0 if _state["start_Book2"] is None else float(_state["start_Book2"]),
        "end_Book2":   -1.0 if _state["end_Book2"]   is None else float(_state["end_Book2"]),
        "start_Book3": -1.0 if _state["start_Book3"] is None else float(_state["start_Book3"]),
        "end_Book3":   -1.0 if _state["end_Book3"]   is None else float(_state["end_Book3"]),
        "updated_at": _time.time(),
    }
    tmp = OUT_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, OUT_PATH)

def mark_attempt():
    _state["overall_attempts_total"] += 1
    _write_times()

def mark_correct(book_name: str, elapsed: float = None):
    key = book_name.strip()
    if key in ("Book1", "Book2", "Book3") and _state[key] is None:
        # elapsed comes from main loop (end - start); fallback to rel
        if elapsed is None:
            elapsed = _rel()
        _state[key] = round(elapsed, 3)
        print(f"{key}: {_state[key]}s", flush=True)
        _write_times()

def mark_finished():
    _state["overall_time_s"] = round(_rel(), 3)
    _write_times()
    print(
        f"overall_attempts_total = {_state['overall_attempts_total']}  overall_time_s = {_state['overall_time_s']}",
        flush=True,
    )
# ============================================================================ #

# ffmpeg -f avfoundation -list_devices true -i ""
AVF_DEVICE = "0:"      # iPhone Continuity Camera device
FPS = 30
CANDIDATE_SIZES = [(640, 480)]
INPUT_PIXFMTS = ["bgr0", "nv12", "yuyv422", "uyvy422"]

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
DETECTOR   = cv2.aruco.ArucoDetector(ARUCO_DICT, cv2.aruco.DetectorParameters())

LOCATION_IDS = {1: "left", 2: "right", 6: "center"}
OBJECT_IDS   = {3: "Book1", 4: "Book2", 5: "Book3"}
TARGET_ZONE  = {"Book1": "left", "Book2": "center", "Book3": "right"}

HIST_LEN        = 10
START_SPEED_PX  = 1.3
START_FRAMES    = 3
STILL_SPEED_PX  = 0.5
STILL_FRAMES    = 8
ROI_MARGIN      = 50

def now_ts():
    return datetime.now().isoformat(timespec='milliseconds')

def log(msg):
    print(f"[{now_ts()}] {msg}", flush=True)

def terminal_quit_watcher(stop_flag: dict):
    for line in sys.stdin:
        if line.strip().lower() == "q":
            stop_flag["stop"] = True
            break

def ffmpeg_cmd(w, h, fps=FPS, in_pixfmt="bgr0"):
    return (f'ffmpeg -hide_banner -loglevel error -f avfoundation '
            f'-pixel_format {in_pixfmt} -framerate {fps} -video_size {w}x{h} '
            f'-i "{AVF_DEVICE}" -f rawvideo -pix_fmt bgr24 -')

def start_ffmpeg_pipe(w, h):
    return subprocess.Popen(
        shlex.split(ffmpeg_cmd(w, h)),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )

def read_frame_from_pipe(proc, w, h, timeout_s=3.0, first=False):
    bytes_needed = w * h * 3
    buf = b''; start = time.time()
    while len(buf) < bytes_needed:
        chunk = proc.stdout.read(bytes_needed - len(buf))
        if not chunk:
            return None
        buf += chunk
        if first and (time.time() - start) > timeout_s and len(buf) < bytes_needed:
            return None
    return np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 3))

def try_open_stream():
    for (w, h) in CANDIDATE_SIZES:
        log(f"Starting FFmpeg capture from device '{AVF_DEVICE}' at {w}x{h}@{FPS}")
        proc = start_ffmpeg_pipe(w, h)
        frame = read_frame_from_pipe(proc, w, h, timeout_s=3.0, first=True)
        if frame is not None:
            return proc, w, h
        try: proc.terminate()
        except Exception: pass
        time.sleep(0.2)
    raise RuntimeError("Could not start FFmpeg stream.")

def detect_aruco(gray):
    return DETECTOR.detectMarkers(gray)

def center_of_corners(corners):
    pts = corners[0]; c = pts.mean(axis=0)
    return float(c[0]), float(c[1])

def rect_from_corners(corners, margin=0):
    pts = corners[0]
    x1, y1 = pts.min(axis=0);  x2, y2 = pts.max(axis=0)
    return (int(x1 - margin), int(y1 - margin), int(x2 + margin), int(y2 + margin))

def point_in_rect(pt, rect):
    x, y = pt; x1, y1, x2, y2 = rect
    return (x1 <= x <= x2) and (y1 <= y <= y2)

def rect_center(rect):
    x1, y1, x2, y2 = rect
    return int((x1+x2)//2), int((y1+y2)//2)

def avg_speed_px(history):
    if len(history) < 2: return 0.0
    d = 0.0
    for i in range(1, len(history)):
        _, x0, y0 = history[i-1]; _, x1, y1 = history[i]
        dx = x1 - x0; dy = y1 - y0
        d += (dx*dx + dy*dy) ** 0.5
    return d / (len(history) - 1)

def median_center(history):
    xs = [x for _, x, _ in history]; ys = [y for _, _, y in history]
    return float(np.median(xs)), float(np.median(ys))

def calibrate_zones(proc, W, H, stop_flag):
    zones = {}
    log("Show location tags: ID 1 (left), ID 2 (right), ID 6 (center)… (press q in terminal to quit)")
    cv2.namedWindow("calibration", cv2.WINDOW_NORMAL)
    while not stop_flag["stop"]:
        frame_ro = read_frame_from_pipe(proc, W, H)
        if frame_ro is None:
            raise RuntimeError("FFmpeg stream ended.")
        frame = frame_ro.copy()
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detect_aruco(gray)

        if ids is not None:
            for i, m_id in enumerate(ids.flatten()):
                if m_id in LOCATION_IDS and LOCATION_IDS[m_id] not in zones:
                    rect = rect_from_corners(corners[i], margin=ROI_MARGIN)
                    zones[LOCATION_IDS[m_id]] = rect
                    cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (0,255,255), 2)
                    cv2.putText(frame, f"{LOCATION_IDS[m_id]} (ID {m_id})",
                                (rect[0], rect[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        cv2.putText(frame, f"Calibrated: {list(zones.keys())}", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.imshow("calibration", frame); cv2.waitKey(1)

        if all(k in zones for k in ("left", "right", "center")):
            cv2.destroyWindow("calibration")
            for name in ("left", "center", "right"):
                rect = zones[name]; cx, cy = rect_center(rect)
                log(f"[CAL] {name:6s} rect {rect}, center=({cx},{cy})")
            return zones

    raise SystemExit("Quit from terminal during calibration.")

# ---------- Main ----------
def run():
    global _trial_t0
    _trial_t0 = _time.perf_counter()  # trial start (DON'T reset again)
    # fresh trial state
    for k in ("Book1","Book2","Book3"):
        _state[k] = None
    _state["overall_attempts_total"] = 0
    _state["overall_time_s"] = 0.0
    for k in ("start_Book1","end_Book1","start_Book2","end_Book2","start_Book3","end_Book3"):
        _state[k] = None

    print(f"[times] saving to: {OUT_PATH}", flush=True)
    _write_times()

    stop_flag = {"stop": False}
    threading.Thread(target=terminal_quit_watcher, args=(stop_flag,), daemon=True).start()

    proc, W, H = try_open_stream()
    zones = calibrate_zones(proc, W, H, stop_flag)

    histories  = {oid: deque(maxlen=HIST_LEN) for oid in OBJECT_IDS}
    remaining  = set(OBJECT_IDS.keys())  # {3,4,5}

    timer_start = {3: None, 4: None, 5: None}  # wall-clock epoch seconds
    result = {
        "Book1": {"attempts": 0, "time_s": None, "done": False, "finish_abs": None},
        "Book2": {"attempts": 0, "time_s": None, "done": False, "finish_abs": None},
        "Book3": {"attempts": 0, "time_s": None, "done": False, "finish_abs": None},
    }
    total_start = None
    last_event = {3: "none", 4: "none", 5: "none"}

    cv2.namedWindow("placement", cv2.WINDOW_NORMAL)

    try:
        while remaining and not stop_flag["stop"]:
            frame_ro = read_frame_from_pipe(proc, W, H)
            if frame_ro is None:
                try: proc.terminate()
                except Exception: pass
                proc, W, H = try_open_stream()
                continue

            frame = frame_ro.copy()
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detect_aruco(gray)

            # draw zones
            for name, rect in zones.items():
                cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (50,180,50), 2)
                cv2.putText(frame, name, (rect[0], rect[1]-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,180,50), 2)

            # track object centers
            if ids is not None:
                for i, m_id in enumerate(ids.flatten()):
                    if m_id in OBJECT_IDS:
                        cx, cy = center_of_corners(corners[i])
                        histories[m_id].append((time.time(), cx, cy))
                        cv2.circle(frame, (int(cx), int(cy)), 4, (0,200,255), -1)
                        cv2.putText(frame, OBJECT_IDS[m_id], (int(cx)+8, int(cy)-8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)

            # handle each object
            for oid in (3,4,5):
                name = OBJECT_IDS[oid]
                if result[name]["done"]:
                    continue
                hist = histories[oid]

                # start overall timer on first movement
                if total_start is None and len(hist) > 2 and avg_speed_px(hist) > START_SPEED_PX:
                    total_start = time.time()
                    # DO NOT reset _trial_t0 here — we want book start/end relative to the true trial start

                # start this object's timer at first movement
                if timer_start[oid] is None:
                    if len(hist) >= START_FRAMES and avg_speed_px(hist) > START_SPEED_PX:
                        timer_start[oid] = time.time()
                        # record relative start once
                        key = f"start_{name}"
                        if _state[key] is None:
                            _state[key] = round(_rel(), 3)
                            _write_times()
                        log(f"[TIMER] {name} timer started (start={_state[key]}s rel).")

                # classify on stillness
                if timer_start[oid] is not None and len(hist) >= STILL_FRAMES and avg_speed_px(hist) < STILL_SPEED_PX:
                    cx, cy = median_center(hist)
                    in_zone = None
                    for zname, rect in zones.items():
                        if point_in_rect((cx, cy), rect):
                            in_zone = zname; break
                    if in_zone is None:
                        last_event[oid] = "none"
                        continue

                    expected = TARGET_ZONE[name]

                    if in_zone != expected:
                        event_tag = f"wrong-{in_zone}"
                        if last_event[oid] != event_tag:
                            result[name]["attempts"] += 1
                            log(f"{name} still in {in_zone} — incorrect, move to {expected}")
                            print("n", flush=True)
                            last_event[oid] = event_tag
                            mark_attempt()
                    else:
                        elapsed = time.time() - timer_start[oid]
                        result[name]["time_s"] = elapsed
                        result[name]["done"]   = True
                        result[name]["finish_abs"] = time.time()
                        remaining.discard(oid)

                        # record relative end time and completion duration
                        _state[f"end_{name}"] = round(_rel(), 3)
                        _write_times()  # flush end immediately

                        log(f"{name} placed in {in_zone} — correct. time_s={round(elapsed,3)} (start={_state[f'start_{name}']} end={_state[f'end_{name}']})")

                        last_event[oid] = "none"
                        mark_attempt()
                        mark_correct(name, elapsed)

            rem_list = [OBJECT_IDS[o] for o in sorted(list(remaining))]
            cv2.putText(frame, f"Remaining: {rem_list}", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.imshow("placement", frame)
            cv2.waitKey(1)

        if stop_flag["stop"]:
            log("Stopped by user (terminal).")
            return

        # -------- Results + Accuracy --------
        if total_start is not None:
            finishes = [result[n]["finish_abs"] for n in ("Book1","Book2","Book3")
                        if result[n]["finish_abs"] is not None]
            total_time = max(finishes) - total_start if finishes else None
        else:
            total_time = None

        log("=== RESULTS ===")
        for nm in ("Book1", "Book2", "Book3"):
            r = result[nm]
            start_r = _state[f"start_{nm}"]
            end_r   = _state[f"end_{nm}"]
            log(f"{nm}: attempts={r['attempts']}, time_s={None if r['time_s'] is None else round(r['time_s'],3)}, start={start_r}, end={end_r}")

        # final stamp of overall & fill any missing per-book fields
        total_attempts = sum(r["attempts"] + 1 for r in result.values())
        _state["overall_attempts_total"] = total_attempts
        _state["overall_time_s"] = -1.0 if total_time is None else float(round(total_time, 3))

        for nm in ("Book1","Book2","Book3"):
            if _state[nm] is None and result[nm]["time_s"] is not None:
                _state[nm] = float(round(result[nm]["time_s"], 3))
            # if we somehow never stamped start/end but have time_s, try to infer end
            if _state[f"end_{nm}"] is None and _state[nm] is not None and _state[f"start_{nm}"] is not None:
                _state[f"end_{nm}"] = round(_state[f"start_{nm}"] + _state[nm], 3)

        _write_times()
        print(f"[times] wrote {OUT_PATH}", flush=True)

        # Accuracy logs
        N = 3
        per_object_attempts = {}
        for nm in ("Book1", "Book2", "Book3"):
            wrong = result[nm]["attempts"]
            attempts = wrong + 1
            per_object_attempts[nm] = attempts
            acc = 100.0 * (1.0 / attempts)
            log(f"{nm}: attempts_total={attempts}, accuracy_percent={acc:.1f}")

        overall_acc = 100.0 * (N / total_attempts)
        log(f"overall_attempts_total = {total_attempts}, overall_time_s = {None if total_time is None else round(total_time,3)}")
        log(f"overall_accuracy_percent = {overall_acc:.1f}")

    finally:
        try: proc.terminate()
        except Exception: pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run()