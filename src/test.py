# ================== IMPORT ==================
import os
import time
import csv
import queue
import argparse
import threading
from threading import Lock

import serial

import numpy as np
import cv2
import imutils
import tensorflow as tf

from imutils.video import VideoStream
from flask import Flask, jsonify, render_template_string, request, Response

import facenet
import align.detect_face
from sklearn.svm import SVC
import pickle

# ================== ARGPARSE ==================
parser = argparse.ArgumentParser()
parser.add_argument("--port", default="COM3", help="Serial port (e.g., COM3)")
parser.add_argument("--baud", default=115200, type=int, help="Serial baudrate")
parser.add_argument("--cam", default="0", help="Camera index (0/1/2) or video path")
args = parser.parse_args()

SERIAL_PORT = args.port
BAUDRATE = args.baud
try:
    CAM_SRC = int(args.cam) if str(args.cam).isdigit() else args.cam
except:
    CAM_SRC = args.cam

# ================== SERIAL CONFIG ==================
try:
    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
    print(f"[SERIAL] Connected to {SERIAL_PORT}")
except Exception as e:
    print(f"[SERIAL] ERROR: {e}")
    ser = None

# ================== GLOBAL STATE ==================
state = {
    "temp": None,
    "hum": None,
    "pir": 0,
    "mq2": 0,
    "ldr": 0,
    "rain_ao": 0,
    "rain_do": 0,

    "rfid": "NONE",
    "rfid_valid": 0,
    "rfid_name": "",

    "relay": 0,
    "led": 0,
    "servo": 0,

    "face_name": "None",
    "face_prob": 0.0,

    "camera_on": 0,
    "alert": ""
}
state_lock = Lock()

# ================== THRESHOLDS ==================
MQ2_THRESHOLD = 2000
LDR_DARK_THRESHOLD = 2000          # >2000 là tối
RAIN_AO_RAIN_THRESHOLD = 3000      # <=3000 là mưa

# ================== MANUAL OVERRIDE ==================
manual_led_until = 0
MANUAL_LED_HOLD_SEC = 120

manual_fan_until = 0
MANUAL_FAN_HOLD_SEC = 30

manual_door_until = 0
MANUAL_DOOR_HOLD_SEC = 10

# ================== SERIAL TX QUEUE + ANTI-SPAM ==================
tx_queue = queue.Queue(maxsize=300)
_last_cmd_time = {}
CMD_COOLDOWN = {
    "LED_ON": 0.35,
    "LED_OFF": 0.35,
    "RELAY_ON": 0.60,
    "RELAY_OFF": 0.60,
    "SERVO_OPEN": 1.50,
    "SERVO_CLOSE": 1.00
}
DEFAULT_COOLDOWN = 0.20

def send_cmd(cmd: str):
    global ser, _last_cmd_time
    if ser is None:
        return
    now = time.time()
    cd = CMD_COOLDOWN.get(cmd, DEFAULT_COOLDOWN)
    last = _last_cmd_time.get(cmd, 0.0)
    if (now - last) < cd:
        return
    _last_cmd_time[cmd] = now
    try:
        tx_queue.put_nowait(cmd)
        print("[SEND]", cmd)
    except queue.Full:
        print("[SEND] TX queue FULL -> drop:", cmd)

def serial_writer():
    global ser
    if ser is None:
        print("[SERIAL] Not connected, skip serial_writer.")
        return
    while True:
        cmd = tx_queue.get()
        try:
            ser.write((cmd + "\n").encode())
        except Exception as e:
            print("[SERIAL WRITER] Error:", e)
            time.sleep(0.2)

# ================== RFID CSV STORAGE ==================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "rfid_cards.csv")

cards_lock = Lock()
allowed_cards = {}

def normalize_uid(uid: str) -> str:
    return (uid or "").strip().upper().replace(" ", "")

def ensure_csv_exists():
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["uid", "name", "created_at"])
        print("[RFID] Created CSV:", CSV_PATH)

def load_cards():
    global allowed_cards
    ensure_csv_exists()
    tmp = {}
    try:
        with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                uid = normalize_uid(row.get("uid", ""))
                name = (row.get("name", "") or "").strip()
                if uid:
                    tmp[uid] = name
    except Exception as e:
        print("[RFID] load_cards error:", e)
    with cards_lock:
        allowed_cards = tmp
    print(f"[RFID] Loaded {len(tmp)} cards")

def save_card(uid: str, name: str) -> bool:
    uid = normalize_uid(uid)
    name = (name or "").strip()
    if not uid:
        return False
    ensure_csv_exists()
    with cards_lock:
        allowed_cards[uid] = name
        try:
            with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["uid", "name", "created_at"])
                for k, v in allowed_cards.items():
                    w.writerow([k, v, time.strftime("%Y-%m-%d %H:%M:%S")])
            return True
        except Exception as e:
            print("[RFID] save_card error:", e)
            return False

def delete_card(uid: str) -> bool:
    uid = normalize_uid(uid)
    if not uid:
        return False
    with cards_lock:
        if uid not in allowed_cards:
            return False
        del allowed_cards[uid]
        try:
            with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["uid", "name", "created_at"])
                for k, v in allowed_cards.items():
                    w.writerow([k, v, time.strftime("%Y-%m-%d %H:%M:%S")])
            return True
        except Exception as e:
            print("[RFID] delete_card error:", e)
            return False

def is_uid_allowed(uid: str):
    uid = normalize_uid(uid)
    if not uid or uid == "NONE":
        return (False, "")
    with cards_lock:
        if uid in allowed_cards:
            return (True, allowed_cards.get(uid, ""))
    return (False, "")

load_cards()

# ================== CAMERA FRAME SHARING FOR WEB ==================
frame_lock = Lock()
latest_frame_bgr = None  # latest BGR frame for MJPEG

def set_latest_frame(frame_bgr):
    global latest_frame_bgr
    with frame_lock:
        latest_frame_bgr = frame_bgr

def gen_mjpeg():
    """MJPEG stream generator from latest_frame_bgr. If camera off, send a placeholder frame."""
    while True:
        with state_lock:
            cam_on = int(state.get("camera_on", 0))

        if cam_on == 0:
            ph = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(ph, "Camera is OFF", (170, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            ok, jpeg = cv2.imencode(".jpg", ph, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
            time.sleep(0.2)
            continue

        with frame_lock:
            frame = None if latest_frame_bgr is None else latest_frame_bgr.copy()

        if frame is None:
            time.sleep(0.03)
            continue

        ok, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            time.sleep(0.03)
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
        time.sleep(0.03)

# ================== RFID anti-freeze processing ==================
last_uid_seen = "NONE"
last_uid_process_time = 0.0
RFID_COOLDOWN_SEC = 1.0

# ================== THREAD 1: SERIAL READER ==================
def serial_reader():
    global last_uid_seen, last_uid_process_time, manual_led_until, manual_fan_until, manual_door_until
    if ser is None:
        print("[SERIAL] Not connected, skip serial_reader.")
        return

    last_led_cmd = None
    last_fan_cmd = None
    gas_triggered = False
    last_is_raining = None
    last_pir = 0

    while True:
        try:
            line = ser.readline().decode(errors="ignore").strip()
            if not line:
                continue

            # ---------- PARSE ----------
            if line.startswith("TEMP:"):
                try:
                    with state_lock: state["temp"] = float(line.split(":")[1])
                except: pass

            elif line.startswith("HUM:"):
                try:
                    with state_lock: state["hum"] = float(line.split(":")[1])
                except: pass

            elif line.startswith("PIR:"):
                try:
                    with state_lock: state["pir"] = int(line.split(":")[1])
                except: pass

            elif line.startswith("MQ2:"):
                try:
                    with state_lock: state["mq2"] = int(line.split(":")[1])
                except: pass

            elif line.startswith("LDR:"):
                try:
                    with state_lock: state["ldr"] = int(line.split(":")[1])
                except: pass

            elif line.startswith("RAIN_AO:"):
                try:
                    with state_lock: state["rain_ao"] = int(line.split(":")[1])
                except: pass

            elif line.startswith("RAIN_DO:"):
                try:
                    with state_lock: state["rain_do"] = int(line.split(":")[1])
                except: pass

            elif line.startswith("RFID:"):
                try:
                    uid = normalize_uid(line.split(":", 1)[1])
                    with state_lock: state["rfid"] = uid
                except: pass

            now = time.time()
            with state_lock:
                mq2_val = int(state["mq2"] or 0)
                ldr_val = int(state["ldr"] or 0)
                rain_ao_val = int(state["rain_ao"] or 0)
                pir_val = int(state["pir"] or 0)
                uid_val = normalize_uid(state["rfid"] or "NONE")

            # ---------- PIR ALERT ----------
            if pir_val == 1 and last_pir == 0:
                with state_lock:
                    state["alert"] = "PHÁT HIỆN CHUYỂN ĐỘNG!"
            last_pir = pir_val

            # ---------- RAIN ALERT ----------
            is_raining = (rain_ao_val <= RAIN_AO_RAIN_THRESHOLD)
            if last_is_raining is None:
                last_is_raining = is_raining
            else:
                if is_raining and (last_is_raining is False):
                    with state_lock:
                        state["alert"] = "CẢNH BÁO MƯA!"
                last_is_raining = is_raining

            # ---------- GAS LOGIC ----------
            if mq2_val > MQ2_THRESHOLD:
                if not gas_triggered:
                    gas_triggered = True
                    with state_lock:
                        state["alert"] = "PHÁT HIỆN KHÍ GAS!"
                if now >= manual_fan_until:
                    if last_fan_cmd != "RELAY_ON":
                        send_cmd("RELAY_ON")
                        with state_lock: state["relay"] = 1
                        last_fan_cmd = "RELAY_ON"
            else:
                gas_triggered = False
                if now >= manual_fan_until:
                    if last_fan_cmd != "RELAY_OFF":
                        send_cmd("RELAY_OFF")
                        with state_lock: state["relay"] = 0
                        last_fan_cmd = "RELAY_OFF"

            # ---------- LDR AUTO LED ----------
            if now >= manual_led_until:
                if ldr_val > LDR_DARK_THRESHOLD:  # tối
                    if last_led_cmd != "LED_ON":
                        send_cmd("LED_ON")
                        with state_lock: state["led"] = 1
                        last_led_cmd = "LED_ON"
                else:  # sáng
                    if last_led_cmd != "LED_OFF":
                        send_cmd("LED_OFF")
                        with state_lock: state["led"] = 0
                        last_led_cmd = "LED_OFF"

            # ---------- RFID AUTH (fix treo) ----------
            if uid_val == "NONE":
                last_uid_seen = "NONE"
                with state_lock:
                    state["rfid_valid"] = 0
                    state["rfid_name"] = ""
                continue

            should_process = False
            if uid_val != last_uid_seen:
                should_process = True
            else:
                if (now - last_uid_process_time) >= RFID_COOLDOWN_SEC:
                    should_process = True

            if should_process:
                last_uid_seen = uid_val
                last_uid_process_time = now

                ok, name = is_uid_allowed(uid_val)
                with state_lock:
                    state["rfid_valid"] = 1 if ok else 0
                    state["rfid_name"] = name if ok else ""

                if ok:
                    # RFID hợp lệ -> mở cửa
                    if now >= manual_door_until:
                        send_cmd("SERVO_OPEN")
                        with state_lock:
                            state["servo"] = 100
                            state["alert"] = f"RFID OK: {name or uid_val} → Mở cửa!"
                else:
                    with state_lock:
                        state["alert"] = f"RFID SAI: {uid_val} (Không mở cửa)"

        except Exception as e:
            print("[SERIAL] Error:", e)
            time.sleep(0.1)

# ================== FACE THREAD CONTROLLER (Start/Stop Camera) ==================
face_thread_obj = None
face_stop_event = threading.Event()
face_running_lock = Lock()

def is_face_running():
    with face_running_lock:
        return (face_thread_obj is not None) and face_thread_obj.is_alive()

def start_face():
    global face_thread_obj
    if is_face_running():
        return True
    face_stop_event.clear()
    face_thread_obj = threading.Thread(target=face_worker, daemon=True)
    face_thread_obj.start()
    return True

def stop_face():
    face_stop_event.set()
    with state_lock:
        state["camera_on"] = 0
        state["face_name"] = "None"
        state["face_prob"] = 0.0
    return True

# ================== THREAD 2: FACE WORKER ==================
def face_worker():
    global manual_door_until

    BASE_DIR2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CLASSIFIER_PATH = os.path.join(BASE_DIR2, 'Models', 'facemodel.pkl')
    FACENET_MODEL_PATH = os.path.join(BASE_DIR2, 'Models', '20180402-114759.pb')

    MINSIZE = 20
    THRESHOLD = [0.6, 0.7, 0.7]
    FACTOR = 0.709
    INPUT_IMAGE_SIZE = 160

    # ---- Robust load classifier (fix invalid load key, 'v') ----
    try:
        with open(CLASSIFIER_PATH, "rb") as f:
            head = f.read(64)
            f.seek(0)

            # Git LFS pointer detection
            if head.startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise RuntimeError(
                    "facemodel.pkl đang là Git LFS pointer (file giả). "
                    "Hãy chạy: git lfs install && git lfs pull"
                )

            # Try normal pickle
            try:
                model, class_names = pickle.load(f)
            except Exception:
                # fallback for old pickle encoding
                f.seek(0)
                model, class_names = pickle.load(f, encoding="latin1")

        print("[FACE] Classifier loaded OK:", len(class_names), "classes")

    except Exception as e:
        print("[FACE] ERROR load classifier:", e)
        with state_lock:
            state["alert"] = f"FACE ERROR: {e}"
            state["camera_on"] = 0
        return

    # ---- Start tensorflow graph ----
    try:
        with tf.Graph().as_default():
            gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
            sess = tf.compat.v1.Session(
                config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False)
            )

            with sess.as_default():
                print('[FACE] Loading FaceNet model...')
                facenet.load_model(FACENET_MODEL_PATH)

                images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")

                pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "src/align")

                # open camera
                with state_lock:
                    state["camera_on"] = 1
                    state["alert"] = "Camera started"

                cap = VideoStream(src=CAM_SRC).start()
                time.sleep(0.8)

                verified_start = 0.0
                opened_by_face = False
                FACE_OPEN_AFTER_SEC = 2.0

                while not face_stop_event.is_set():
                    frame = cap.read()
                    if frame is None:
                        time.sleep(0.02)
                        continue

                    frame = imutils.resize(frame, width=720)
                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    bounding_boxes, _ = align.detect_face.detect_face(
                        rgb, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR
                    )

                    faces_found = bounding_boxes.shape[0]
                    name_show = "None"
                    prob_show = 0.0
                    face_ok = False

                    if faces_found == 1:
                        det = bounding_boxes[:, 0:4]
                        bb = np.zeros((faces_found, 4), dtype=np.int32)
                        bb[0][0] = det[0][0]
                        bb[0][1] = det[0][1]
                        bb[0][2] = det[0][2]
                        bb[0][3] = det[0][3]

                        if (bb[0][3] - bb[0][1]) / frame.shape[0] > 0.25:
                            cropped = frame[bb[0][1]:bb[0][3], bb[0][0]:bb[0][2], :]
                            scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                                interpolation=cv2.INTER_CUBIC)
                            scaled = facenet.prewhiten(scaled)
                            scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)

                            feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                            emb_array = sess.run(embeddings, feed_dict=feed_dict)

                            predictions = model.predict_proba(emb_array)
                            best_class_indices = np.argmax(predictions, axis=1)
                            best_class_probabilities = predictions[
                                np.arange(len(best_class_indices)), best_class_indices
                            ]
                            best_name = class_names[best_class_indices[0]]
                            prob = float(best_class_probabilities[0])

                            prob_show = prob
                            if prob > 0.8:
                                face_ok = True
                                name_show = best_name
                            else:
                                face_ok = False
                                name_show = "Unknown"

                            color = (0, 255, 0) if face_ok else (0, 0, 255)
                            cv2.rectangle(frame, (bb[0][0], bb[0][1]), (bb[0][2], bb[0][3]), color, 2)
                            cv2.putText(frame, f"{name_show} {prob_show:.3f}",
                                        (bb[0][0], bb[0][3] + 28),
                                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.1, (255, 255, 255), 2, 2)

                    elif faces_found > 1:
                        cv2.putText(frame, "Only one face", (14, 34),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                        face_ok = False
                        name_show = "Multiple"
                        prob_show = 0.0

                    # push to web
                    set_latest_frame(frame)

                    with state_lock:
                        state["face_name"] = name_show
                        state["face_prob"] = float(prob_show)

                    now = time.time()

                    # Face OK -> open door after 2s continuous
                    if face_ok:
                        if verified_start == 0:
                            verified_start = now
                            opened_by_face = False
                        else:
                            if (now - verified_start) >= FACE_OPEN_AFTER_SEC and not opened_by_face:
                                if now >= manual_door_until:
                                    send_cmd("SERVO_OPEN")
                                    with state_lock:
                                        state["servo"] = 100
                                        state["alert"] = f"FACE OK: {name_show} → Mở cửa!"
                                opened_by_face = True
                    else:
                        verified_start = 0
                        opened_by_face = False
                        if name_show == "Unknown":
                            with state_lock:
                                state["alert"] = "PHÁT HIỆN NGƯỜI LẠ!"

                # stop
                cap.stop()
                with state_lock:
                    state["camera_on"] = 0
                    state["alert"] = "Camera stopped"

    except Exception as e:
        print("[FACE] ERROR runtime:", e)
        with state_lock:
            state["camera_on"] = 0
            state["alert"] = f"FACE ERROR: {e}"

# ================== WEB (FLASK) ==================
app = Flask(__name__)

HTML_PAGE = r"""
<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"/>
<title>Smart Home</title>
<style>
:root{
  --bg:#0b0c10; --bg2:#111218;
  --stroke:rgba(255,255,255,.08);
  --text:#f5f6fa; --sub:rgba(245,246,250,.72); --muted:rgba(245,246,250,.55);
  --green:#30d158; --blue:#0a84ff; --orange:#ff9f0a; --red:#ff453a;
  --shadow:0 10px 30px rgba(0,0,0,.35);
  --radius:22px; --radius2:16px;
}
*{box-sizing:border-box}
body{
  margin:0;
  background: radial-gradient(1200px 600px at 30% -10%, rgba(10,132,255,.18), transparent 60%),
              radial-gradient(900px 500px at 90% 10%, rgba(48,209,88,.14), transparent 55%),
              linear-gradient(180deg, var(--bg), var(--bg2));
  color:var(--text);
  font-family:-apple-system,BlinkMacSystemFont,"SF Pro Display","SF Pro Text",Segoe UI,Roboto,Arial,sans-serif;
  -webkit-font-smoothing:antialiased;
  padding: env(safe-area-inset-top) 14px calc(env(safe-area-inset-bottom) + 70px);
}
.app{max-width:430px;margin:0 auto;padding:6px 4px 24px}
.topbar{display:flex;align-items:flex-end;justify-content:space-between;padding:8px 8px 10px}
.title{font-size:28px;font-weight:900;letter-spacing:-.02em;line-height:1.05}
.subtitle{font-size:13px;color:var(--muted);display:flex;align-items:center;gap:8px;margin-top:6px}
.dot{width:8px;height:8px;border-radius:99px;background:var(--green);box-shadow:0 0 0 4px rgba(48,209,88,.14)}
.pill{display:inline-flex;align-items:center;gap:8px;padding:8px 10px;border-radius:999px;background:rgba(255,255,255,.06);border:1px solid var(--stroke);box-shadow:0 10px 25px rgba(0,0,0,.25)}
.pill small{color:var(--sub);font-weight:700}
.pill .time{color:var(--text);font-weight:900;font-size:12px;letter-spacing:.02em}

.card{background:linear-gradient(180deg,rgba(255,255,255,.06),rgba(255,255,255,.04));
  border:1px solid var(--stroke);border-radius:var(--radius);box-shadow:var(--shadow);
  padding:14px;margin:10px 8px;backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px)}
.card h3{margin:0 0 10px;font-size:14px;color:var(--sub);font-weight:900;letter-spacing:.02em;text-transform:uppercase}

.grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.tile{background:rgba(0,0,0,.18);border:1px solid rgba(255,255,255,.08);border-radius:var(--radius2);
  padding:12px 12px 10px;display:flex;flex-direction:column;gap:8px;position:relative;overflow:hidden}
.row{display:flex;align-items:center;justify-content:space-between;gap:10px;position:relative;z-index:1}
.label{font-size:13px;color:var(--sub);font-weight:900}
.value{font-size:22px;font-weight:900;letter-spacing:-.02em}
.unit{font-size:12px;color:var(--muted);font-weight:900;margin-left:4px}
.hint{font-size:12px;color:var(--muted);font-weight:700}

.device{display:flex;align-items:center;justify-content:space-between;gap:12px;background:rgba(0,0,0,.18);
  border:1px solid rgba(255,255,255,.08);border-radius:var(--radius2);padding:12px}
.devleft{display:flex;gap:12px;align-items:center;min-width:0}
.icon{width:44px;height:44px;border-radius:14px;display:grid;place-items:center;border:1px solid rgba(255,255,255,.10);
  background:rgba(255,255,255,.06);box-shadow:0 10px 20px rgba(0,0,0,.28);flex:0 0 auto}
.devmeta{display:flex;flex-direction:column;gap:2px;min-width:0}
.devname{font-weight:900;font-size:16px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.devdesc{font-size:12px;color:var(--muted);font-weight:750;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}

.switch{position:relative;width:54px;height:32px;flex:0 0 auto}
.switch input{display:none}
.slider{position:absolute;inset:0;background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.10);
  border-radius:999px;transition:.2s ease;box-shadow:inset 0 0 0 1px rgba(0,0,0,.25)}
.slider:before{content:"";position:absolute;height:26px;width:26px;left:3px;top:2.5px;background:rgba(255,255,255,.95);
  border-radius:999px;box-shadow:0 10px 18px rgba(0,0,0,.35);transition:.2s ease}
.switch input:checked + .slider{background:rgba(48,209,88,.95);border-color:rgba(48,209,88,.7);box-shadow:0 10px 22px rgba(48,209,88,.18)}
.switch input:checked + .slider:before{transform:translateX(22px);background:#fff}

.btn{border:0;cursor:pointer;padding:12px 14px;border-radius:16px;font-weight:900;letter-spacing:.01em;
  background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.10);color:var(--text);
  box-shadow:0 10px 20px rgba(0,0,0,.25);user-select:none}
.btn:active{transform:scale(.98)}
.btn.primary{background:linear-gradient(180deg, rgba(10,132,255,.92), rgba(10,132,255,.72));border-color:rgba(10,132,255,.35)}
.btn.danger{background:linear-gradient(180deg, rgba(255,69,58,.92), rgba(255,69,58,.72));border-color:rgba(255,69,58,.35)}
.btn.good{background:linear-gradient(180deg, rgba(48,209,88,.92), rgba(48,209,88,.72));border-color:rgba(48,209,88,.35)}

.chip{padding:8px 10px;border-radius:999px;font-weight:900;font-size:12px;border:1px solid rgba(255,255,255,.10);
  background:rgba(255,255,255,.06);color:var(--text);flex:0 0 auto}
.chip.good{background:rgba(48,209,88,.16);border-color:rgba(48,209,88,.30)}
.chip.bad{background:rgba(255,69,58,.16);border-color:rgba(255,69,58,.30)}
.chip.warn{background:rgba(255,159,10,.16);border-color:rgba(255,159,10,.30)}

.video{
  width:100%;
  aspect-ratio: 16/9;
  background:#000;
  border-radius: 18px;
  border:1px solid rgba(255,255,255,.08);
  overflow:hidden;
}
.video img{width:100%;height:100%;object-fit:cover;display:block}

pre{margin:0;padding:10px 10px;background:rgba(0,0,0,.18);border:1px solid rgba(255,255,255,.08);
  border-radius:16px;overflow:auto;color:rgba(245,246,250,.85);font-size:12px;line-height:1.35;
  font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,"Liberation Mono",monospace}

.tabbar{
  position:fixed;left:50%;transform:translateX(-50%);
  bottom: env(safe-area-inset-bottom);
  width:min(430px, calc(100% - 16px));
  background:rgba(20,21,27,.86);
  border:1px solid rgba(255,255,255,.10);
  border-radius:22px;
  box-shadow:0 18px 40px rgba(0,0,0,.45);
  backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);
  display:flex;justify-content:space-around;gap:8px;
  padding:10px;
  z-index:9999;
}
.tabbar button{
  flex:1;border:0;background:transparent;color:rgba(245,246,250,.65);
  font-weight:900;padding:10px;border-radius:16px;cursor:pointer
}
.tabbar button.active{background:rgba(255,255,255,.10);color:var(--text)}
.toast{position:fixed;left:50%;transform:translateX(-50%);bottom:calc(env(safe-area-inset-bottom) + 90px);
  width:min(430px, calc(100% - 24px));z-index:99999;display:none}
.toast .inner{background:rgba(20,21,27,.92);border:1px solid rgba(255,255,255,.10);border-radius:18px;
  padding:12px 14px;box-shadow:0 18px 40px rgba(0,0,0,.45);backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);
  display:flex;align-items:flex-start;gap:10px}
.toast .mark{width:28px;height:28px;border-radius:10px;display:grid;place-items:center;background:rgba(255,69,58,.18);
  border:1px solid rgba(255,69,58,.25);flex:0 0 auto;margin-top:1px}
.toast .text{font-weight:900;color:var(--text);line-height:1.25;font-size:14px}
.toast .sub{margin-top:3px;font-size:12px;color:var(--muted);font-weight:750}

table{width:100%;border-collapse:separate;border-spacing:0 8px}
td,th{font-size:12px;color:rgba(245,246,250,.85);text-align:left;padding:10px}
th{color:var(--muted);font-weight:900;text-transform:uppercase;font-size:11px;letter-spacing:.04em}
.tr{background:rgba(0,0,0,.18);border:1px solid rgba(255,255,255,.08);border-radius:14px}
.tr td:first-child{border-top-left-radius:14px;border-bottom-left-radius:14px}
.tr td:last-child{border-top-right-radius:14px;border-bottom-right-radius:14px}
.inp{width:100%; padding:12px 12px; border-radius:14px;background:rgba(0,0,0,.18); border:1px solid rgba(255,255,255,.10);
  color:var(--text); outline:none; font-weight:900;}
.row2{display:flex;gap:10px}
</style>
</head>

<body>
<div class="app">
  <div class="topbar">
    <div>
      <div class="title">Smart Home</div>
      <div class="subtitle">
        <span class="dot" id="connDot"></span>
        <span id="connText">Connected</span>
        <span style="opacity:.35;">•</span>
        <span id="lastUpdate">--:--:--</span>
      </div>
    </div>
    <div class="pill">
      <small>Home</small>
      <span class="time" id="clock">--:--</span>
    </div>
  </div>

  <!-- OVERVIEW -->
  <section id="viewOverview">
    <div class="card">
      <h3>Environment</h3>
      <div class="grid">
        <div class="tile">
          <div class="row"><div class="label">Temperature</div><div class="value"><span id="temp">--</span><span class="unit">°C</span></div></div>
          <div class="hint">DHT11</div>
        </div>
        <div class="tile">
          <div class="row"><div class="label">Humidity</div><div class="value"><span id="hum">--</span><span class="unit">%</span></div></div>
          <div class="hint">DHT11</div>
        </div>
        <div class="tile">
          <div class="row"><div class="label">Gas (MQ-2)</div><div class="value"><span id="mq2">--</span></div></div>
          <div class="hint" id="mq2Hint">Safe</div>
        </div>
        <div class="tile">
          <div class="row"><div class="label">Light (LDR)</div><div class="value"><span id="ldr">--</span></div></div>
          <div class="hint" id="ldrHint">Bright</div>
        </div>
        <div class="tile">
          <div class="row"><div class="label">Rain (AO)</div><div class="value"><span id="rainAO">--</span></div></div>
          <div class="hint" id="rainHint">Sunny</div>
        </div>
        <div class="tile">
          <div class="row"><div class="label">Motion (PIR)</div><div class="value"><span id="pir">--</span></div></div>
          <div class="hint" id="pirHint">No motion</div>
        </div>
      </div>
    </div>

    <div class="card">
      <h3>Access</h3>

      <div class="device">
        <div class="devleft">
          <div class="icon">RF</div>
          <div class="devmeta">
            <div class="devname" id="rfidTitle">RFID</div>
            <div class="devdesc" id="rfid">NONE</div>
          </div>
        </div>
        <span class="chip" id="rfidChip">Ready</span>
      </div>

      <div style="height:10px"></div>

      <div class="device">
        <div class="devleft">
          <div class="icon">🚪</div>
          <div class="devmeta">
            <div class="devname">Door</div>
            <div class="devdesc" id="doorDesc">Closed</div>
          </div>
        </div>
        <div style="display:flex; gap:8px; align-items:center;">
          <button class="btn primary" onclick="fetch('/api/door/open',{method:'POST'})">Open</button>
          <button class="btn" onclick="fetch('/api/door/close',{method:'POST'})">Close</button>
        </div>
      </div>
    </div>
  </section>

  <!-- DEVICES -->
  <section id="viewDevices" style="display:none;">
    <div class="card">
      <h3>Devices</h3>

      <div class="device">
        <div class="devleft">
          <div class="icon">💡</div>
          <div class="devmeta">
            <div class="devname">Light</div>
            <div class="devdesc" id="ledDesc">Off</div>
          </div>
        </div>
        <label class="switch">
          <input type="checkbox" id="ledToggle" onchange="toggleDevice('led', this.checked)">
          <span class="slider"></span>
        </label>
      </div>

      <div style="height:10px"></div>

      <div class="device">
        <div class="devleft">
          <div class="icon">🌀</div>
          <div class="devmeta">
            <div class="devname">Fan</div>
            <div class="devdesc" id="fanDesc">Off</div>
          </div>
        </div>
        <label class="switch">
          <input type="checkbox" id="fanToggle" onchange="toggleDevice('fan', this.checked)">
          <span class="slider"></span>
        </label>
      </div>
    </div>
  </section>

  <!-- CAMERA -->
  <section id="viewCamera" style="display:none;">
    <div class="card">
      <h3>Camera & Face</h3>
      <div class="video">
        <img id="camImg" src="/video_feed" />
      </div>
      <div style="height:10px"></div>

      <div class="device">
        <div class="devleft">
          <div class="devmeta">
            <div class="devname" id="camStatusTitle">Camera</div>
            <div class="devdesc" id="camStatusDesc">OFF</div>
          </div>
        </div>
        <span class="chip" id="camChip">OFF</span>
      </div>

      <div style="height:10px"></div>

      <div class="device">
        <div class="devleft">
          <div class="devmeta">
            <div class="devname" id="faceName">Face: None</div>
            <div class="devdesc" id="faceProb">Prob: 0.000</div>
          </div>
        </div>
        <span class="chip" id="faceChip">Idle</span>
      </div>

      <div style="height:12px"></div>

      <div style="display:flex; gap:10px;">
        <button class="btn good" style="flex:1" onclick="camStart()">Start Camera</button>
        <button class="btn danger" style="flex:1" onclick="camStop()">Stop Camera</button>
      </div>

      <div style="height:10px;color:var(--muted);font-weight:800;font-size:12px;line-height:1.45;">
        • Camera chỉ mở khi bạn bấm Start. Face nhận diện đúng (prob &gt; 0.8, giữ 2s) sẽ mở cửa.
      </div>
    </div>
  </section>

  <!-- CARDS -->
  <section id="viewCards" style="display:none;">
    <div class="card">
      <h3>Register RFID card</h3>
      <div class="row2">
        <input class="inp" id="cardName" placeholder="Name (e.g., Son)" />
        <button class="btn primary" style="white-space:nowrap" onclick="useLastUID()">Use last scanned</button>
      </div>
      <div style="height:10px"></div>
      <input class="inp" id="cardUID" placeholder="UID (e.g., 04A1B2C3)" />

      <div style="height:10px"></div>
      <div class="row2">
        <button class="btn primary" style="flex:1" onclick="registerCard()">Save to CSV</button>
        <button class="btn danger" style="flex:1" onclick="reloadCards()">Refresh list</button>
      </div>

      <div style="height:12px"></div>

      <div class="device">
        <div class="devleft">
          <div class="devmeta">
            <div class="devname">CSV file</div>
            <div class="devdesc" id="csvPath"></div>
          </div>
        </div>
        <span class="chip" id="cardCountChip">0 cards</span>
      </div>

      <div style="height:12px"></div>
      <div style="overflow:auto;">
        <table>
          <thead>
            <tr><th>UID</th><th>Name</th><th>Action</th></tr>
          </thead>
          <tbody id="cardsTbody"></tbody>
        </table>
      </div>
    </div>
  </section>

  <!-- DETAILS -->
  <section id="viewDetails" style="display:none;">
    <div class="card">
      <h3>Raw status</h3>
      <pre id="statusRaw">{}</pre>
    </div>
  </section>
</div>

<div class="tabbar">
  <button class="active" id="tabOverview" onclick="setTab('overview')">Overview</button>
  <button id="tabDevices" onclick="setTab('devices')">Devices</button>
  <button id="tabCamera" onclick="setTab('camera')">Camera</button>
  <button id="tabCards" onclick="setTab('cards')">Cards</button>
  <button id="tabDetails" onclick="setTab('details')">Details</button>
</div>

<div class="toast" id="toast">
  <div class="inner">
    <div class="mark" aria-hidden="true">!</div>
    <div>
      <div class="text" id="toastText">Alert</div>
      <div class="sub" id="toastSub">Just now</div>
    </div>
  </div>
</div>

<script>
let lastAlert = '';
let lastUID = "NONE";

function setTab(tab){
  const tabs = ['overview','devices','camera','cards','details'];
  for(const t of tabs){
    document.getElementById('view'+t.charAt(0).toUpperCase()+t.slice(1)).style.display = (tab===t) ? 'block' : 'none';
    document.getElementById('tab'+t.charAt(0).toUpperCase()+t.slice(1)).classList.toggle('active', tab===t);
  }
  if(tab==='cards') reloadCards();
}

function showToast(msg){
  const toast = document.getElementById('toast');
  document.getElementById('toastText').textContent = msg;
  document.getElementById('toastSub').textContent = new Date().toLocaleTimeString();
  toast.style.display = 'block';
  clearTimeout(window.__toastTimer);
  window.__toastTimer = setTimeout(()=>{ toast.style.display='none'; }, 3200);
}

function updateClock(){
  const d = new Date();
  document.getElementById('clock').textContent = d.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'});
}
setInterval(updateClock, 1000);
updateClock();

async function toggleDevice(type, on){
  if(type==='led'){
    await fetch(on ? '/api/led/on' : '/api/led/off', {method:'POST'});
  }else if(type==='fan'){
    await fetch(on ? '/api/fan/on' : '/api/fan/off', {method:'POST'});
  }
}

async function camStart(){
  const r = await fetch('/api/cam/start', {method:'POST'});
  const j = await r.json();
  showToast(j.message || "Camera start");
  document.getElementById('camImg').src = '/video_feed?t=' + Date.now();
}
async function camStop(){
  const r = await fetch('/api/cam/stop', {method:'POST'});
  const j = await r.json();
  showToast(j.message || "Camera stop");
}

function useLastUID(){
  document.getElementById('cardUID').value = (lastUID && lastUID !== "NONE") ? lastUID : "";
  if(!document.getElementById('cardUID').value){
    showToast("No scanned UID yet. Scan a card first.");
  }
}

async function registerCard(){
  const name = document.getElementById('cardName').value.trim();
  const uid  = document.getElementById('cardUID').value.trim().toUpperCase().replaceAll(" ","");
  if(!uid){ showToast("UID is empty"); return; }
  const r = await fetch('/api/rfid/register', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({uid, name})});
  const j = await r.json();
  showToast(j.message || "Saved");
  reloadCards();
}

async function reloadCards(){
  const r = await fetch('/api/rfid/list', {cache:'no-store'});
  const j = await r.json();
  document.getElementById('csvPath').textContent = j.csv_path || "";
  document.getElementById('cardCountChip').textContent = (j.cards ? j.cards.length : 0) + " cards";

  const tb = document.getElementById('cardsTbody');
  tb.innerHTML = "";
  (j.cards || []).forEach(c=>{
    const tr = document.createElement('tr');
    tr.className = 'tr';
    tr.innerHTML = `
      <td>${c.uid}</td>
      <td>${c.name || ''}</td>
      <td><button class="btn danger" style="padding:8px 10px;border-radius:14px" onclick="deleteCard('${c.uid}')">Delete</button></td>
    `;
    tb.appendChild(tr);
  });
}

async function deleteCard(uid){
  const r = await fetch('/api/rfid/delete/' + encodeURIComponent(uid), {method:'POST'});
  const j = await r.json();
  showToast(j.message || "Deleted");
  reloadCards();
}

async function updateStatus(){
  try{
    const r = await fetch('/status', {cache:"no-store"});
    const d = await r.json();

    document.getElementById('connDot').style.background = 'var(--green)';
    document.getElementById('connText').textContent = 'Connected';
    document.getElementById('lastUpdate').textContent = new Date().toLocaleTimeString();

    document.getElementById('temp').textContent = (d.temp===null) ? "--" : Number(d.temp).toFixed(1);
    document.getElementById('hum').textContent  = (d.hum===null)  ? "--" : Number(d.hum).toFixed(1);
    document.getElementById('mq2').textContent  = (d.mq2===null)  ? "--" : d.mq2;
    document.getElementById('ldr').textContent  = (d.ldr===null)  ? "--" : d.ldr;
    document.getElementById('rainAO').textContent = (d.rain_ao===null) ? "--" : d.rain_ao;
    document.getElementById('pir').textContent  = (d.pir===null) ? "--" : d.pir;

    const mq2Val = Number(d.mq2 || 0);
    const ldrVal = Number(d.ldr || 0);
    const rainAO = Number(d.rain_ao || 0);
    const pirVal = Number(d.pir || 0);

    document.getElementById('mq2Hint').textContent = mq2Val > 2000 ? "Danger" : "Safe";
    document.getElementById('ldrHint').textContent = (ldrVal > 2000) ? "Dark" : "Bright";
    document.getElementById('rainHint').textContent = (rainAO <= 3000) ? "Raining" : "Sunny";
    document.getElementById('pirHint').textContent = pirVal === 1 ? "Motion detected" : "No motion";

    // RFID
    lastUID = (d.rfid || "NONE").toUpperCase();
    document.getElementById('rfid').textContent = lastUID;

    const chip = document.getElementById('rfidChip');
    const title = document.getElementById('rfidTitle');

    if(d.rfid_valid === 1){
      chip.textContent = "Valid";
      chip.className = "chip good";
      title.textContent = "RFID (Valid)";
    }else if(lastUID !== "NONE"){
      chip.textContent = "Invalid";
      chip.className = "chip bad";
      title.textContent = "RFID (Invalid)";
    }else{
      chip.textContent = "Ready";
      chip.className = "chip";
      title.textContent = "RFID";
    }

    // Door text
    const servoPos = Number(d.servo || 0);
    document.getElementById('doorDesc').textContent = servoPos > 0 ? "Open" : "Closed";

    // toggles
    const ledOn = Number(d.led || 0) === 1;
    const fanOn = Number(d.relay || 0) === 1;
    document.getElementById('ledToggle').checked = ledOn;
    document.getElementById('fanToggle').checked = fanOn;
    document.getElementById('ledDesc').textContent = ledOn ? "On" : "Off";
    document.getElementById('fanDesc').textContent = fanOn ? "On" : "Off";

    // camera status
    const camOn = Number(d.camera_on || 0) === 1;
    document.getElementById('camStatusDesc').textContent = camOn ? "ON" : "OFF";
    const camChip = document.getElementById('camChip');
    camChip.textContent = camOn ? "ON" : "OFF";
    camChip.className = camOn ? "chip good" : "chip";

    // face
    document.getElementById('faceName').textContent = "Face: " + (d.face_name || "None");
    document.getElementById('faceProb').textContent = "Prob: " + (Number(d.face_prob||0).toFixed(3));
    const faceChip = document.getElementById('faceChip');
    if((d.face_name||"").toLowerCase() === "unknown"){
      faceChip.textContent = "Unknown";
      faceChip.className = "chip bad";
    }else if((d.face_name||"none").toLowerCase() !== "none"){
      faceChip.textContent = "Detected";
      faceChip.className = "chip warn";
    }else{
      faceChip.textContent = "Idle";
      faceChip.className = "chip";
    }

    document.getElementById('statusRaw').textContent = JSON.stringify(d, null, 2);

    if(d.alert && d.alert !== lastAlert){
      lastAlert = d.alert;
      showToast(d.alert);
    }

  }catch(e){
    document.getElementById('connDot').style.background = 'var(--red)';
    document.getElementById('connText').textContent = 'Disconnected';
  }
}

setInterval(updateStatus, 1000);
updateStatus();
</script>
</body>
</html>
"""

# ================== FLASK ROUTES ==================
@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

@app.route("/status")
def get_status():
    with state_lock:
        return jsonify(dict(state))

@app.route("/video_feed")
def video_feed():
    return Response(gen_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")

# --- Manual controls ---
@app.route("/api/door/open", methods=["POST"])
def api_door_open():
    global manual_door_until
    manual_door_until = time.time() + MANUAL_DOOR_HOLD_SEC
    send_cmd("SERVO_OPEN")
    with state_lock:
        state["servo"] = 100
        state["alert"] = "Mở cửa thủ công!"
    return jsonify({"ok": True})

@app.route("/api/door/close", methods=["POST"])
def api_door_close():
    global manual_door_until
    manual_door_until = time.time() + MANUAL_DOOR_HOLD_SEC
    send_cmd("SERVO_CLOSE")
    with state_lock:
        state["servo"] = 0
        state["alert"] = "Đóng cửa thủ công!"
    return jsonify({"ok": True})

@app.route("/api/led/on", methods=["POST"])
def api_led_on():
    global manual_led_until
    manual_led_until = time.time() + MANUAL_LED_HOLD_SEC
    send_cmd("LED_ON")
    with state_lock:
        state["led"] = 1
        state["alert"] = "Bật đèn thủ công!"
    return jsonify({"ok": True})

@app.route("/api/led/off", methods=["POST"])
def api_led_off():
    global manual_led_until
    manual_led_until = time.time() + MANUAL_LED_HOLD_SEC
    send_cmd("LED_OFF")
    with state_lock:
        state["led"] = 0
        state["alert"] = "Tắt đèn thủ công!"
    return jsonify({"ok": True})

@app.route("/api/fan/on", methods=["POST"])
def api_fan_on():
    global manual_fan_until
    manual_fan_until = time.time() + MANUAL_FAN_HOLD_SEC
    send_cmd("RELAY_ON")
    with state_lock:
        state["relay"] = 1
        state["alert"] = "Bật quạt thủ công!"
    return jsonify({"ok": True})

@app.route("/api/fan/off", methods=["POST"])
def api_fan_off():
    global manual_fan_until
    manual_fan_until = time.time() + MANUAL_FAN_HOLD_SEC
    send_cmd("RELAY_OFF")
    with state_lock:
        state["relay"] = 0
        state["alert"] = "Tắt quạt thủ công!"
    return jsonify({"ok": True})

# --- Camera start/stop commands ---
@app.route("/api/cam/start", methods=["POST"])
def api_cam_start():
    ok = start_face()
    with state_lock:
        state["alert"] = "Start camera command sent"
    return jsonify({"ok": ok, "message": "Camera starting..."})

@app.route("/api/cam/stop", methods=["POST"])
def api_cam_stop():
    ok = stop_face()
    with state_lock:
        state["alert"] = "Stop camera command sent"
    return jsonify({"ok": ok, "message": "Camera stopped"})

# --- RFID APIs ---
@app.route("/api/rfid/list")
def api_rfid_list():
    with cards_lock:
        cards = [{"uid": k, "name": v} for k, v in sorted(allowed_cards.items())]
    return jsonify({"csv_path": CSV_PATH, "cards": cards})

@app.route("/api/rfid/register", methods=["POST"])
def api_rfid_register():
    data = request.get_json(silent=True) or {}
    uid = normalize_uid(data.get("uid", ""))
    name = (data.get("name", "") or "").strip()

    if not uid:
        return jsonify({"ok": False, "message": "UID rỗng"}), 400

    ok = save_card(uid, name)
    if ok:
        with state_lock:
            state["alert"] = f"Đã lưu thẻ: {uid} ({name})"
        return jsonify({"ok": True, "message": f"Saved: {uid}"})
    return jsonify({"ok": False, "message": "Không lưu được CSV"}), 500

@app.route("/api/rfid/delete/<uid>", methods=["POST"])
def api_rfid_delete(uid):
    uid = normalize_uid(uid)
    ok = delete_card(uid)
    if ok:
        with state_lock:
            state["alert"] = f"Đã xóa thẻ: {uid}"
        return jsonify({"ok": True, "message": f"Deleted: {uid}"})
    return jsonify({"ok": False, "message": "UID không tồn tại"}), 404

# ================== MAIN ==================
if __name__ == "__main__":
    t_writer = threading.Thread(target=serial_writer, daemon=True)
    t_writer.start()

    t_serial = threading.Thread(target=serial_reader, daemon=True)
    t_serial.start()

    # camera/face thread only starts when /api/cam/start
    with state_lock:
        state["camera_on"] = 0

    app.run(host="0.0.0.0", port=5000, debug=False)
