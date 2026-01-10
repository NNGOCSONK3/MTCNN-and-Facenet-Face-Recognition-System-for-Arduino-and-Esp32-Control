# ================== IMPORT ==================
import os
import time
import csv
import queue
import argparse
import threading
from threading import Lock
import logging 

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

# ================== SYSTEM CONFIGURATION ==================
sys_config = {
    "serial_port": args.port,
    "baudrate": args.baud,
    "cam_src": args.cam,
    "mq2_thresh": 2000,
    "ldr_thresh": 900, 
    "rain_thresh": 3000,
    "door_auto_close_sec": 5
}
config_lock = Lock()

cam_restart_flag = False

# ================== SERIAL CONFIG ==================
ser = None

def reconnect_serial(port, baud):
    global ser
    try:
        if ser and ser.is_open:
            ser.close()
        ser = serial.Serial(port, baud, timeout=1)
        print(f"[SERIAL] Connected to {port} @ {baud}")
        return True, "Connected"
    except Exception as e:
        print(f"[SERIAL] Connection Error: {e}")
        ser = None
        return False, str(e)

reconnect_serial(sys_config["serial_port"], sys_config["baudrate"])

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
    "door_open_ts": 0,

    "face_name": "None",
    "face_prob": 0.0,

    "camera_on": 0,
    "alert": "",
    
    # LCD State
    "lcd_line1": "System Init...",
    
    "is_capturing": False,
    "capture_progress": 0,
    "capture_target": 30,
    "capture_name": "",
    "is_training": False
}
state_lock = Lock()

# ================== SERIAL TX QUEUE ==================
tx_queue = queue.Queue(maxsize=300)
_last_cmd_time = {}
CMD_COOLDOWN = {
    "LED_ON": 0.2, "LED_OFF": 0.2,
    "RELAY_ON": 0.5, "RELAY_OFF": 0.5,
    "SERVO_OPEN": 1.0, "SERVO_CLOSE": 1.0,
    "LCD": 0.5 
}
DEFAULT_COOLDOWN = 0.20

def send_cmd(cmd: str):
    global ser, _last_cmd_time
    if ser is None: return
    
    key = cmd.split(":")[0] if ":" in cmd else cmd
    now = time.time()
    cd = CMD_COOLDOWN.get(key, DEFAULT_COOLDOWN)
    last = _last_cmd_time.get(key, 0.0)
    
    if (now - last) < cd: return
    _last_cmd_time[key] = now
    
    try:
        tx_queue.put_nowait(cmd)
        if not cmd.startswith("LCD:"):
            print("[SEND]", cmd)
    except queue.Full:
        pass

def serial_writer():
    global ser
    while True:
        if ser is None:
            time.sleep(1)
            continue
        try:
            cmd = tx_queue.get()
            ser.write((cmd + "\n").encode())
        except Exception as e:
            print("[SERIAL WRITER] Error:", e)
            time.sleep(0.2)

# ================== FILES & PATHS ==================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "rfid_cards.csv")
LOGO_PATH = os.path.join(BASE_DIR, "smart_home_logo.txt")
DATASET_PATH = os.path.join(BASE_DIR, "dataset")

if not os.path.exists(DATASET_PATH): os.makedirs(DATASET_PATH)

logo_base64 = ""
try:
    if os.path.exists(LOGO_PATH):
        with open(LOGO_PATH, "r", encoding="utf-8") as f:
            logo_base64 = f.read().strip()
except Exception: pass

# ================== RFID CSV STORAGE ==================
cards_lock = Lock()
allowed_cards = {}

def normalize_uid(uid: str) -> str:
    return (uid or "").strip().upper().replace(" ", "")

def ensure_csv_exists():
    if not os.path.exists(CSV_PATH):
        try:
            with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["uid", "name", "created_at"])
        except: pass

def load_cards():
    global allowed_cards
    ensure_csv_exists()
    tmp = {}
    try:
        if os.path.exists(CSV_PATH):
            with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
                r = csv.DictReader(f)
                for row in r:
                    uid = normalize_uid(row.get("uid", ""))
                    name = (row.get("name", "") or "").strip()
                    if uid: tmp[uid] = name
    except: pass
    with cards_lock: allowed_cards = tmp

def save_card(uid: str, name: str) -> bool:
    uid = normalize_uid(uid)
    name = (name or "").strip()
    if not uid: return False
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
        except: return False

def delete_card(uid: str) -> bool:
    uid = normalize_uid(uid)
    if not uid: return False
    with cards_lock:
        if uid not in allowed_cards: return False
        del allowed_cards[uid]
        try:
            with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["uid", "name", "created_at"])
                for k, v in allowed_cards.items():
                    w.writerow([k, v, time.strftime("%Y-%m-%d %H:%M:%S")])
            return True
        except: return False

def is_uid_allowed(uid: str):
    uid = normalize_uid(uid)
    if not uid or uid == "NONE": return (False, "")
    with cards_lock:
        if uid in allowed_cards: return (True, allowed_cards.get(uid, ""))
    return (False, "")

load_cards()

# ================== CAMERA FRAME SHARING ==================
frame_lock = Lock()
latest_frame_bgr = None

def set_latest_frame(frame_bgr):
    global latest_frame_bgr
    with frame_lock: latest_frame_bgr = frame_bgr

def gen_mjpeg():
    while True:
        with state_lock: cam_on = int(state.get("camera_on", 0))
        if cam_on == 0:
            ph = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(ph, "Camera is OFF", (170, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            ok, jpeg = cv2.imencode(".jpg", ph, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok: yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
            time.sleep(0.2)
            continue

        with frame_lock: frame = None if latest_frame_bgr is None else latest_frame_bgr.copy()
        if frame is None:
            time.sleep(0.03)
            continue
        ok, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            time.sleep(0.03)
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
        time.sleep(0.03)

# ================== HELPERS ==================
def send_lcd_update(force_msg=None):
    with state_lock:
        l1 = force_msg if force_msg else state.get("lcd_line1", "Smart Home")
        t = state.get("temp")
        h = state.get("hum")
    
    if force_msg:
        with state_lock: state["lcd_line1"] = force_msg

    l2 = "Wait Sensors..."
    if t is not None and h is not None:
        l2 = f"T:{t:.1f}C H:{h:.0f}%"
    
    send_cmd(f"LCD:{l1}#{l2}")

def open_door_logic(trigger_source):
    now = time.time()
    send_cmd("SERVO_OPEN")
    send_lcd_update("Door Open")
    with state_lock:
        state["servo"] = 100
        state["door_open_ts"] = now
        state["alert"] = f"{trigger_source} -> Door Open!"

# ================== FACE THREAD CONTROLLER ==================
# Đặt ở đây để đảm bảo biến được khởi tạo trước khi dùng
face_thread_obj = None
face_stop_event = threading.Event()
face_train_trigger = False # Flag trigger training
face_running_lock = Lock()

def is_face_running():
    with face_running_lock:
        return (face_thread_obj is not None) and face_thread_obj.is_alive()

# ================== THREAD 1: SERIAL READER & LOGIC ==================
def serial_reader():
    global last_uid_seen, last_uid_process_time
    
    last_uid_seen = "NONE"
    last_uid_process_time = 0.0
    RFID_COOLDOWN_SEC = 1.0
    last_led_cmd = None
    last_fan_cmd = None
    gas_triggered = False
    last_is_raining = None
    last_pir = 0
    last_lcd_ts = 0
    
    # Timer cho LDR (để tránh nhiễu khi đọc Analog)
    last_light_check_ts = 0

    while True:
        if ser is None: 
            time.sleep(1)
            continue

        try:
            line = ser.readline().decode(errors="ignore").strip()
            if line:
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
                door_state = int(state["servo"] or 0)
                door_ts = state["door_open_ts"]

            with config_lock:
                TH_MQ2 = sys_config["mq2_thresh"]
                TH_LDR = sys_config["ldr_thresh"]
                TH_RAIN = sys_config["rain_thresh"]
                AUTO_CLOSE = sys_config["door_auto_close_sec"]

            # 1. LCD UPDATE LOOP
            if now - last_lcd_ts > 2.0:
                send_lcd_update()
                last_lcd_ts = now

            # 2. AUTO DOOR CLOSE
            if door_state > 0 and (now - door_ts > AUTO_CLOSE):
                send_cmd("SERVO_CLOSE")
                send_lcd_update("Door Closed")
                with state_lock:
                    state["servo"] = 0
                    state["alert"] = "Door Auto-Closed"

            # 3. SENSORS
            # PIR
            if pir_val == 1 and last_pir == 0:
                with state_lock: state["alert"] = "MOTION DETECTED!"
            last_pir = pir_val

            # RAIN
            is_raining = (rain_ao_val <= TH_RAIN)
            if is_raining and (last_is_raining is False):
                with state_lock: state["alert"] = "RAIN ALERT!"
                send_lcd_update("Rain Alert!")
            last_is_raining = is_raining

            # GAS LOGIC
            if mq2_val > TH_MQ2:
                if not gas_triggered:
                    gas_triggered = True
                    with state_lock: state["alert"] = "GAS DETECTED!"
                    send_lcd_update("GAS DANGER!")
                
                if last_fan_cmd != "RELAY_ON":
                    send_cmd("RELAY_ON")
                    last_fan_cmd = "RELAY_ON"
                    with state_lock: state["relay"] = 1
            else:
                if gas_triggered:
                    gas_triggered = False
                    send_lcd_update("System Ready")
                    send_cmd("RELAY_OFF")
                    last_fan_cmd = "RELAY_OFF"
                    with state_lock: state["relay"] = 0

            # LDR LOGIC (Analog Mode with Hysteresis)
            if now - last_light_check_ts > 2.0:
                last_light_check_ts = now
                
                # Logic mới: Vượt ngưỡng cài đặt là bật luôn (chỉ giữ trễ khi tắt)
                if ldr_val > TH_LDR:
                    if last_led_cmd != "LED_ON":
                        print(f"[AUTO] DARK ({ldr_val} > {TH_LDR}) -> LED ON")
                        send_cmd("LED_ON")
                        with state_lock: state["led"] = 1
                        last_led_cmd = "LED_ON"
                
                # Tắt đèn khi sáng hơn ngưỡng - 100
                elif ldr_val < (TH_LDR - 100):
                    if last_led_cmd != "LED_OFF":
                        print(f"[AUTO] BRIGHT ({ldr_val} < {TH_LDR-100}) -> LED OFF")
                        send_cmd("LED_OFF")
                        with state_lock: state["led"] = 0
                        last_led_cmd = "LED_OFF"

            # 4. RFID
            if uid_val == "NONE":
                last_uid_seen = "NONE"
                with state_lock:
                    state["rfid_valid"] = 0
                    state["rfid_name"] = ""
                continue

            should_process = False
            if uid_val != last_uid_seen: should_process = True
            else:
                if (now - last_uid_process_time) >= RFID_COOLDOWN_SEC: should_process = True

            if should_process:
                last_uid_seen = uid_val
                last_uid_process_time = now
                ok, name = is_uid_allowed(uid_val)
                with state_lock:
                    state["rfid_valid"] = 1 if ok else 0
                    state["rfid_name"] = name if ok else ""

                if ok:
                    open_door_logic(f"RFID ({name})")
                else:
                    with state_lock: state["alert"] = f"INVALID RFID: {uid_val}"
                    send_lcd_update("Invalid Card")

        except Exception as e:
            print("[SERIAL] Error:", e)
            time.sleep(0.1)

# ================== THREAD 2: FACE WORKER ==================
def face_worker():
    global face_train_trigger, cam_restart_flag

    # Paths
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = os.path.dirname(CURRENT_DIR)
    path_models_p = os.path.join(PARENT_DIR, 'Models')
    path_models_c = os.path.join(CURRENT_DIR, 'Models')
    MODELS_DIR = path_models_p if os.path.exists(path_models_p) else path_models_c
    CLASSIFIER_PATH = os.path.join(MODELS_DIR, 'facemodel.pkl')
    FACENET_MODEL_PATH = os.path.join(MODELS_DIR, '20180402-114759.pb')
    path_align_1 = os.path.join(CURRENT_DIR, 'align')       
    path_align_2 = os.path.join(CURRENT_DIR, 'src', 'align') 
    ALIGN_PATH = path_align_1 if os.path.exists(path_align_1) else path_align_2

    if not os.path.exists(ALIGN_PATH):
        print(f"[FACE] ERR: No align folder at {ALIGN_PATH}")
        with state_lock: state["camera_on"] = 0
        return

    MINSIZE = 20
    THRESHOLD = [0.6, 0.7, 0.7]
    FACTOR = 0.709
    INPUT_IMAGE_SIZE = 160

    def load_svc():
        try:
            with open(CLASSIFIER_PATH, "rb") as f:
                head = f.read(64); f.seek(0)
                if head.startswith(b"version"): raise RuntimeError("Git LFS pointer")
                try: return pickle.load(f)
                except: f.seek(0); return pickle.load(f, encoding="latin1")
        except: return None, []

    model, class_names = load_svc()
    if model is None:
        with state_lock: state["camera_on"] = 0; state["alert"] = "Load Model Fail"
        return

    try:
        with tf.Graph().as_default():
            gpu_opts = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
            sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_opts))
            with sess.as_default():
                facenet.load_model(FACENET_MODEL_PATH)
                images_ph = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_ph = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
                pnet, rnet, onet = align.detect_face.create_mtcnn(sess, ALIGN_PATH)

                with state_lock: state["camera_on"] = 1
                with config_lock: 
                    src = sys_config["cam_src"]
                    try: src = int(src)
                    except: pass
                
                cap = VideoStream(src=src).start()
                time.sleep(0.8)
                verified_start = 0
                opened_by_face = False
                FACE_OPEN_AFTER_SEC = 2.0

                while not face_stop_event.is_set():
                    if cam_restart_flag:
                        cap.stop()
                        with config_lock:
                            src = sys_config["cam_src"]
                            try: src = int(src)
                            except: pass
                        cap = VideoStream(src=src).start()
                        time.sleep(1.0)
                        cam_restart_flag = False

                    frame = cap.read()
                    if frame is None: time.sleep(0.02); continue
                    
                    frame = imutils.resize(frame, width=720)
                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # --- TRAINING ---
                    if face_train_trigger:
                        face_train_trigger = False
                        with state_lock: state["is_training"] = True
                        try:
                            dataset_classes = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]
                            emb_array = []
                            labels = []
                            names_map = []
                            nrof = 0
                            for idx, cname in enumerate(dataset_classes):
                                names_map.append(cname)
                                cdir = os.path.join(DATASET_PATH, cname)
                                for fimg in os.listdir(cdir):
                                    if not fimg.lower().endswith(('.jpg','.png')): continue
                                    img = cv2.imread(os.path.join(cdir, fimg))
                                    if img is None: continue
                                    try:
                                        sc = cv2.resize(img, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE))
                                        sc = facenet.prewhiten(cv2.cvtColor(sc, cv2.COLOR_BGR2RGB))
                                        sc = sc.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                                        feed = {images_ph: sc, phase_train_ph: False}
                                        emb = sess.run(embeddings, feed_dict=feed)
                                        emb_array.append(emb[0])
                                        labels.append(idx)
                                        nrof += 1
                                    except: pass
                            
                            if nrof > 0:
                                new_svc = SVC(kernel='linear', probability=True)
                                new_svc.fit(emb_array, labels)
                                with open(CLASSIFIER_PATH, 'wb') as f: pickle.dump((new_svc, names_map), f)
                                model = new_svc
                                class_names = names_map
                                with state_lock: state["alert"] = "Training Done!"
                            else:
                                with state_lock: state["alert"] = "No images to train"
                        except Exception as e:
                            print("Train err:", e)
                        
                        with state_lock: state["is_training"] = False
                        continue

                    # --- DETECT ---
                    with state_lock:
                        capturing = state["is_capturing"]
                        cap_name = state["capture_name"]
                        cap_prog = state["capture_progress"]
                        cap_tgt = state["capture_target"]

                    bounding_boxes, _ = align.detect_face.detect_face(rgb, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)
                    n_faces = bounding_boxes.shape[0]

                    # --- CAPTURE ---
                    if capturing and n_faces == 1:
                        det = bounding_boxes[:,0:4]
                        bb = np.zeros(4, dtype=np.int32)
                        bb[0]=max(det[0][0],0); bb[1]=max(det[0][1],0)
                        bb[2]=min(det[0][2], frame.shape[1]); bb[3]=min(det[0][3], frame.shape[0])
                        
                        if (bb[2]-bb[0]>0) and (bb[3]-bb[1]>0):
                            crop = frame[bb[1]:bb[3], bb[0]:bb[2], :]
                            sdir = os.path.join(DATASET_PATH, cap_name)
                            if not os.path.exists(sdir): os.makedirs(sdir)
                            cv2.imwrite(os.path.join(sdir, f"{int(time.time()*1000)}.jpg"), crop)
                            
                            with state_lock:
                                state["capture_progress"] += 1
                                if state["capture_progress"] >= cap_tgt:
                                    state["is_capturing"] = False
                                    face_train_trigger = True

                    # --- RECOGNIZE ---
                    name_show = "None"
                    prob_show = 0.0
                    face_ok = False

                    if n_faces == 1:
                        det = bounding_boxes[:,0:4]
                        bb = np.zeros(4, dtype=np.int32)
                        bb[0]=det[0][0]; bb[1]=det[0][1]; bb[2]=det[0][2]; bb[3]=det[0][3]
                        
                        if (bb[3]-bb[1])/frame.shape[0] > 0.2:
                            crop = frame[max(0,bb[1]):min(frame.shape[0],bb[3]), max(0,bb[0]):min(frame.shape[1],bb[2]), :]
                            try:
                                sc = cv2.resize(crop, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
                                sc = facenet.prewhiten(sc).reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                                feed = {images_ph: sc, phase_train_ph: False}
                                embs = sess.run(embeddings, feed_dict=feed)
                                preds = model.predict_proba(embs)
                                best = np.argmax(preds, axis=1)[0]
                                prob = preds[0][best]
                                
                                name_show = class_names[best]
                                prob_show = prob
                                if prob > 0.8: face_ok = True
                                else: name_show = "Unknown"
                            except: pass
                            
                            color = (0,255,0) if face_ok else (0,0,255)
                            if capturing: color = (255,255,0)
                            cv2.rectangle(frame, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), color, 2)
                            
                            label = f"{name_show} {prob_show:.2f}"
                            if capturing: label = f"REC: {cap_prog}/{cap_tgt}"
                            cv2.putText(frame, label, (int(bb[0]), int(bb[3])+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

                    set_latest_frame(frame)
                    with state_lock:
                        state["face_name"] = name_show
                        state["face_prob"] = float(prob_show)

                    now = time.time()
                    if face_ok and not capturing:
                        if verified_start == 0: verified_start = now
                        else:
                            if (now - verified_start) >= FACE_OPEN_AFTER_SEC and not opened_by_face:
                                open_door_logic(f"FACE ({name_show})")
                                opened_by_face = True
                    else:
                        verified_start = 0
                        opened_by_face = False
                        if name_show == "Unknown" and not capturing:
                            with state_lock: state["alert"] = "UNKNOWN PERSON DETECTED!"
                            send_lcd_update("Stranger Detect")

                cap.stop()
                with state_lock: state["camera_on"] = 0; state["alert"] = "Cam Stopped"

    except Exception as e:
        print("[FACE] Err:", e)
        with state_lock: state["camera_on"] = 0; state["alert"] = str(e)

# ================== FACE CONTROLLER IMPL ==================
def start_face():
    global face_thread_obj
    if is_face_running(): return True
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

# ================== FLASK WEB ==================
app = Flask(__name__)
# Silence Werkzeug Logger
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

HTML_PAGE = '''
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"/>
<title>Smart Home AI</title>
<style>
:root{
  --primary: #0a84ff; 
  --primary-glow: rgba(10,132,255,0.4);
  --bg: #0f0f13;
  --surface: rgba(255, 255, 255, 0.05);
  --surface-hover: rgba(255, 255, 255, 0.08);
  --border: rgba(255, 255, 255, 0.1);
  --text: #ffffff;
  --text-dim: rgba(255, 255, 255, 0.6);
  --success: #30d158;
  --danger: #ff453a;
  --warning: #ff9f0a;
  --radius: 20px;
  --shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
}

body{
  margin:0;
  background-color: var(--bg);
  background-image: 
    radial-gradient(at 10% 10%, var(--primary-glow) 0px, transparent 50%),
    radial-gradient(at 90% 90%, rgba(48, 209, 88, 0.15) 0px, transparent 50%);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  min-height: 100vh;
  padding-bottom: 90px;
}

*{box-sizing:border-box}
.glass{
  background: var(--surface);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border: 1px solid var(--border);
  box-shadow: var(--shadow);
}
.container{ max-width: 600px; margin: 0 auto; padding: 15px; }

/* HEADER */
.header{
  display: flex; align-items: center; justify-content: space-between;
  padding: 10px 0 20px;
}
.brand{ display: flex; align-items: center; gap: 12px; }
.logo-box{
  width: 50px; height: 50px; border-radius: 12px; overflow: hidden;
  background: rgba(255,255,255,0.1); display: grid; place-items: center;
}
.logo-box img{ width: 100%; height: 100%; object-fit: contain; }
.brand h1{ margin: 0; font-size: 24px; font-weight: 800; line-height: 1.1; }
.brand p{ margin: 0; font-size: 13px; color: var(--text-dim); }
.status-pill{
  padding: 6px 12px; border-radius: 50px; font-size: 12px; font-weight: 700;
  background: rgba(255,255,255,0.05); border: 1px solid var(--border);
  display: flex; align-items: center; gap: 6px;
}
.dot{ width: 8px; height: 8px; background: var(--success); border-radius: 50%; box-shadow: 0 0 8px var(--success); }
.dot.offline{ background: var(--danger); box-shadow: 0 0 8px var(--danger); }

/* GRID LAYOUT */
.grid-2{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.grid-3{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; }

/* CARDS */
.card{
  border-radius: var(--radius); padding: 16px; margin-bottom: 16px;
  position: relative; overflow: hidden; transition: 0.2s;
}
.card-title{
  font-size: 14px; font-weight: 700; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px;
  display: flex; justify-content: space-between; align-items: center;
}

/* SENSORS */
.sensor-box{
  background: rgba(255,255,255,0.03); border-radius: 16px; padding: 12px;
  border: 1px solid rgba(255,255,255,0.05);
  display: flex; flex-direction: column; gap: 4px;
}
.s-val{ font-size: 20px; font-weight: 800; }
.s-label{ font-size: 12px; color: var(--text-dim); }
.s-icon{ font-size: 18px; margin-bottom: 4px; color: var(--primary); }

/* DEVICES */
.device-row{
  display: flex; align-items: center; justify-content: space-between;
  padding: 12px; background: rgba(0,0,0,0.2); border-radius: 16px; margin-bottom: 8px;
}
.dev-info{ display: flex; align-items: center; gap: 12px; }
.dev-icon{ 
  width: 40px; height: 40px; background: rgba(255,255,255,0.1); 
  border-radius: 10px; display: grid; place-items: center; font-size: 20px; 
}

/* CAMERA */
.cam-wrap{
  width: 100%; aspect-ratio: 16/9; background: #000; border-radius: 16px; overflow: hidden;
  position: relative; border: 1px solid var(--border);
}
.cam-wrap img{ width: 100%; height: 100%; object-fit: cover; }
.cam-overlay{
  position: absolute; bottom: 10px; left: 10px; right: 10px;
  display: flex; justify-content: space-between;
}

/* BUTTONS */
.btn{
  border: none; outline: none; padding: 12px; border-radius: 14px;
  font-weight: 700; cursor: pointer; transition: 0.2s;
  background: rgba(255,255,255,0.1); color: white;
  display: flex; align-items: center; justify-content: center; gap: 6px;
}
.btn:active{ transform: scale(0.96); }
.btn-primary{ background: var(--primary); color: white; box-shadow: 0 4px 15px var(--primary-glow); }
.btn-danger{ background: var(--danger); }
.btn-block{ width: 100%; }

/* TOGGLE */
.toggle {
  position: relative; display: inline-block; width: 50px; height: 28px;
}
.toggle input { opacity: 0; width: 0; height: 0; }
.slider {
  position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0;
  background-color: rgba(255,255,255,0.1); transition: .4s; border-radius: 34px;
}
.slider:before {
  position: absolute; content: ""; height: 22px; width: 22px; left: 3px; bottom: 3px;
  background-color: white; transition: .4s; border-radius: 50%;
}
input:checked + .slider { background-color: var(--primary); }
input:checked + .slider:before { transform: translateX(22px); }

/* NAVBAR */
.nav-bar{
  position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%);
  width: 90%; max-width: 450px; height: 65px;
  border-radius: 24px; display: flex; justify-content: space-around; align-items: center;
  z-index: 100;
}
.nav-item{
  background: none; border: none; color: var(--text-dim);
  display: flex; flex-direction: column; align-items: center; gap: 4px;
  font-size: 10px; font-weight: 700; padding: 8px; cursor: pointer;
}
.nav-item.active{ color: white; }
.nav-item.active .nav-icon{ background: rgba(255,255,255,0.15); transform: translateY(-5px); }
.nav-icon{
  width: 36px; height: 36px; border-radius: 12px; display: grid; place-items: center;
  font-size: 18px; transition: 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
}

/* TOAST */
.toast{
  position: fixed; top: 20px; left: 50%; transform: translateX(-50%) translateY(-100px);
  background: var(--surface); backdrop-filter: blur(16px);
  border: 1px solid var(--border); border-radius: 50px;
  padding: 10px 24px; font-weight: 600; font-size: 14px;
  box-shadow: 0 10px 40px rgba(0,0,0,0.5);
  transition: 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275); opacity: 0; z-index: 2000;
  display: flex; align-items: center; gap: 10px;
}
.toast.show{ transform: translateX(-50%) translateY(0); opacity: 1; }

/* TABLE & INPUT */
table { width: 100%; border-collapse: collapse; margin-top: 10px; }
th { text-align: left; color: var(--text-dim); font-size: 12px; padding: 8px; }
td { padding: 12px 8px; border-bottom: 1px solid rgba(255,255,255,0.05); font-size: 14px; }
.inp { 
  background: rgba(0,0,0,0.3); border: 1px solid var(--border); color: white; 
  padding: 12px; border-radius: 12px; width: 100%; outline: none; margin-bottom: 8px;
}
.inp:focus{ border-color: var(--primary); }

.progress-bar {
    width: 100%; height: 10px; background: rgba(255,255,255,0.1);
    border-radius: 5px; margin-top: 10px; overflow: hidden;
}
.progress-fill {
    height: 100%; background: var(--success); width: 0%; transition: width 0.3s ease;
}
.label-input { font-size: 12px; color: var(--text-dim); margin-bottom: 4px; display: block; }
.danger-text { color: var(--danger); animation: blink 1s infinite; }
@keyframes blink { 50% { opacity: 0.5; } }

</style>
</head>
<body>

<div class="container">
  <!-- HEADER -->
  <header class="header">
    <div class="brand">
      <div class="logo-box">
        {% if logo %}
          <img src="data:image/png;base64,{{ logo }}" alt="Logo">
        {% else %}
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color:var(--primary)"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path><polyline points="9 22 9 12 15 12 15 22"></polyline></svg>
        {% endif %}
      </div>
      <div>
        <h1>Smart Home</h1>
        <p id="clock">--:--</p>
      </div>
    </div>
    <div class="status-pill">
      <span class="dot" id="connDot"></span>
      <span id="connText">Connecting...</span>
    </div>
  </header>

  <!-- TAB: HOME -->
  <div id="tab-home" class="tab-content">
    
    <!-- Environment -->
    <div class="card glass">
      <div class="card-title">Environment</div>
      <div class="grid-3">
        <div class="sensor-box">
          <div class="s-icon">🌡️</div>
          <div class="s-val"><span id="temp">--</span>°</div>
          <div class="s-label">Temp</div>
        </div>
        <div class="sensor-box">
          <div class="s-icon">💧</div>
          <div class="s-val"><span id="hum">--</span>%</div>
          <div class="s-label">Humidity</div>
        </div>
        <div class="sensor-box" id="mq2Box">
          <div class="s-icon">🌫️</div>
          <div class="s-val" style="font-size:16px" id="mq2">--</div>
          <div class="s-label">Air Quality</div>
        </div>
        <div class="sensor-box">
          <div class="s-icon">💡</div>
          <div class="s-val" style="font-size:16px" id="ldr">--</div>
          <div class="s-label">Light</div>
        </div>
        <div class="sensor-box">
          <div class="s-icon">🌧️</div>
          <div class="s-val" style="font-size:16px" id="rainAO">--</div>
          <div class="s-label">Weather</div>
        </div>
        <div class="sensor-box">
          <div class="s-icon">🏃</div>
          <div class="s-val" style="font-size:16px" id="pir">--</div>
          <div class="s-label">Motion</div>
        </div>
      </div>
    </div>

    <!-- Security -->
    <div class="card glass">
      <div class="card-title">Security Entry</div>
      <div class="device-row">
        <div class="dev-info">
          <div class="dev-icon" style="color:#ff9f0a">🚪</div>
          <div>
            <div style="font-weight:700">Door Lock</div>
            <div style="font-size:12px; color:var(--text-dim)" id="doorDesc">Closed</div>
          </div>
        </div>
        <div style="display:flex; gap:8px">
          <button class="btn" onclick="apiCall('/api/door/open')">Open</button>
          <button class="btn" onclick="apiCall('/api/door/close')">Lock</button>
        </div>
      </div>

      <div class="device-row">
        <div class="dev-info">
          <div class="dev-icon" style="color:#0a84ff">💳</div>
          <div>
            <div style="font-weight:700" id="rfidTitle">RFID Scanner</div>
            <div style="font-size:12px; color:var(--text-dim)" id="rfid">Waiting...</div>
          </div>
        </div>
        <span class="status-pill" id="rfidChip">Idle</span>
      </div>
    </div>

    <!-- Controls -->
    <div class="card glass">
      <div class="card-title">Quick Controls</div>
      <div class="device-row">
        <div class="dev-info">
          <div class="dev-icon" style="color:#ffd60a">💡</div>
          <div>
            <div style="font-weight:700">Living Room Light</div>
            <div style="font-size:12px; color:var(--text-dim)" id="ledDesc">Off</div>
          </div>
        </div>
        <label class="toggle">
          <input type="checkbox" id="ledToggle" onchange="toggleDevice('led', this.checked)">
          <span class="slider"></span>
        </label>
      </div>

      <div class="device-row">
        <div class="dev-info">
          <div class="dev-icon" style="color:#32d74b">❄️</div>
          <div>
            <div style="font-weight:700">Ventilation Fan</div>
            <div style="font-size:12px; color:var(--text-dim)" id="fanDesc">Off</div>
          </div>
        </div>
        <label class="toggle">
          <input type="checkbox" id="fanToggle" onchange="toggleDevice('fan', this.checked)">
          <span class="slider"></span>
        </label>
      </div>
    </div>
  </div>

  <!-- TAB: CAMERA -->
  <div id="tab-camera" class="tab-content" style="display:none">
    <div class="card glass">
      <div class="card-title">Live Surveillance</div>
      <div class="cam-wrap">
        <img id="camImg" src="/video_feed">
        <div class="cam-overlay">
           <span class="status-pill" id="camChip">OFF</span>
        </div>
      </div>
      <br>
      <div class="grid-2">
        <button class="btn btn-primary" onclick="camStart()">Start Camera</button>
        <button class="btn btn-danger" onclick="camStop()">Stop Camera</button>
      </div>
      <br>
      <div class="device-row">
        <div class="dev-info">
          <div class="dev-icon">👤</div>
          <div>
            <div style="font-weight:700" id="faceName">None</div>
            <div style="font-size:12px; color:var(--text-dim)" id="faceProb">Probability: 0.00</div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- TAB: SETTINGS -->
  <div id="tab-settings" class="tab-content" style="display:none">
    
    <!-- Config -->
    <div class="card glass">
      <div class="card-title">System Configuration</div>
      <div style="margin-bottom: 12px;">
        <span class="label-input">Connectivity (Serial & Camera)</span>
        <div class="grid-2">
            <input class="inp" id="cfgPort" placeholder="COM Port (e.g. COM3)">
            <input class="inp" id="cfgCam" placeholder="Cam Index (e.g. 0)">
        </div>
      </div>
      <div style="margin-bottom: 12px;">
        <span class="label-input">Sensor Thresholds (Gas, Light, Rain)</span>
        <div class="grid-3">
            <input class="inp" id="cfgMq2" placeholder="Gas Limit" type="number">
            <input class="inp" id="cfgLdr" placeholder="Light Limit" type="number">
            <input class="inp" id="cfgRain" placeholder="Rain Limit" type="number">
        </div>
      </div>
      <button class="btn btn-primary btn-block" onclick="saveConfig()">Save & Apply</button>
      
      <div style="margin-top: 16px; border-top: 1px solid var(--border); padding-top:12px;">
        <span class="label-input">Send Message to LCD</span>
        <div style="display:flex; gap:8px">
            <input class="inp" id="lcdMsg" placeholder="Type message..." maxlength="16">
            <button class="btn" onclick="sendLcd()">Send</button>
        </div>
      </div>
    </div>

    <!-- Face Reg -->
    <div class="card glass">
      <div class="card-title">Face Registration</div>
      <div style="margin-bottom: 10px; color: var(--text-dim); font-size: 13px;">
         Enter name and start capture. Face camera until progress completes (30 images).
      </div>
      <div style="display:flex; gap:8px; margin-bottom:12px">
         <input class="inp" id="regFaceName" placeholder="New User Name">
      </div>
      <button class="btn btn-primary btn-block" onclick="startFaceReg()">Start Registration</button>
      <div id="regProgressBox" style="display:none; margin-top:15px;">
        <div style="display:flex; justify-content:space-between; font-size:12px; font-weight:700">
            <span id="regStatus">Capturing...</span>
            <span id="regCount">0/30</span>
        </div>
        <div class="progress-bar">
            <div class="progress-fill" id="regFill"></div>
        </div>
      </div>
    </div>

    <!-- RFID -->
    <div class="card glass">
      <div class="card-title">Card Management</div>
      <div style="display:flex; gap:8px; margin-bottom:12px">
         <input class="inp" id="cardName" placeholder="Owner Name">
         <button class="btn" onclick="useLastUID()">Scan</button>
      </div>
      <input class="inp" id="cardUID" placeholder="UID (e.g. A1 B2 C3 D4)">
      <br><br>
      <button class="btn btn-primary btn-block" onclick="registerCard()">Register Card</button>
      <div style="margin-top:20px; max-height:300px; overflow-y:auto">
        <table>
          <thead><tr><th>UID</th><th>NAME</th><th>ACTION</th></tr></thead>
          <tbody id="cardsTbody"></tbody>
        </table>
      </div>
    </div>
  </div>

  <div style="height:60px"></div>
</div>

<!-- BOTTOM NAV -->
<nav class="nav-bar glass">
  <button class="nav-item active" onclick="switchTab('home', this)">
    <div class="nav-icon">🏠</div> Home
  </button>
  <button class="nav-item" onclick="switchTab('camera', this)">
    <div class="nav-icon">📷</div> Camera
  </button>
  <button class="nav-item" onclick="switchTab('settings', this)">
    <div class="nav-icon">⚙️</div> Settings
  </button>
</nav>

<!-- TOAST -->
<div id="toast" class="toast">
  <span style="font-size:18px">🔔</span> <span id="toastMsg">Notification</span>
</div>

<script>
let lastUID = "NONE";
let currentTab = 'home';

// Load Config on Startup
window.onload = function() {
    loadConfig();
};

function switchTab(tabId, btn){
  document.querySelectorAll('.tab-content').forEach(el => el.style.display = 'none');
  document.getElementById('tab-'+tabId).style.display = 'block';
  
  document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
  btn.classList.add('active');
  currentTab = tabId;

  if(tabId === 'settings'){
      reloadCards();
      loadConfig();
  }
}

function showToast(msg){
  const t = document.getElementById('toast');
  document.getElementById('toastMsg').innerText = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 3000);
}

function updateClock(){
  const now = new Date();
  document.getElementById('clock').innerText = now.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'});
}
setInterval(updateClock, 1000);

// API Helpers
async function apiCall(url, data=null){
    const opt = data ? {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(data)} : {method:'POST'};
    try {
        const r = await fetch(url, opt);
        const j = await r.json();
        showToast(j.message || (j.ok ? "Success" : "Failed"));
        return j;
    } catch(e) {
        showToast("Error: " + e);
        return {ok:false};
    }
}

async function toggleDevice(type, checked){
  const action = checked ? 'on' : 'off';
  await apiCall(`/api/${type}/${action}`);
}

async function camStart(){
  const j = await apiCall('/api/cam/start');
  document.getElementById('camImg').src = '/video_feed?t='+Date.now();
}

async function camStop(){ await apiCall('/api/cam/stop'); }

// Config Logic
async function loadConfig(){
    try {
        const r = await fetch('/api/config/get');
        const j = await r.json();
        document.getElementById('cfgPort').value = j.serial_port;
        document.getElementById('cfgCam').value  = j.cam_src;
        document.getElementById('cfgMq2').value  = j.mq2_thresh;
        document.getElementById('cfgLdr').value  = j.ldr_thresh;
        document.getElementById('cfgRain').value = j.rain_thresh;
    } catch(e) { console.log("Config load err", e); }
}

async function saveConfig(){
    await apiCall('/api/config/update', {
        serial_port: document.getElementById('cfgPort').value,
        cam_src: document.getElementById('cfgCam').value,
        mq2_thresh: Number(document.getElementById('cfgMq2').value),
        ldr_thresh: Number(document.getElementById('cfgLdr').value),
        rain_thresh: Number(document.getElementById('cfgRain').value)
    });
}

async function sendLcd(){
    const msg = document.getElementById('lcdMsg').value;
    if(msg) await apiCall('/api/lcd/send', {message: msg});
}

// Face Reg Logic
async function startFaceReg(){
    const name = document.getElementById('regFaceName').value.trim();
    if(!name) return showToast("Please enter a name!");
    
    const j = await apiCall('/api/face/register', {name: name});
    if(j.ok) document.getElementById('regProgressBox').style.display = 'block';
}

// RFID Logic
function useLastUID(){
  if(lastUID && lastUID !== "NONE"){
    document.getElementById('cardUID').value = lastUID;
  } else {
    showToast("Please swipe card first!");
  }
}

async function registerCard(){
  const uid = document.getElementById('cardUID').value;
  const name = document.getElementById('cardName').value;
  if(uid) {
      await apiCall('/api/rfid/register', {uid, name});
      reloadCards();
  } else showToast("UID missing");
}

async function deleteCard(uid){
  if(confirm("Delete this card?")) {
      await apiCall('/api/rfid/delete/'+encodeURIComponent(uid));
      reloadCards();
  }
}

async function reloadCards(){
  const r = await fetch('/api/rfid/list');
  const j = await r.json();
  const tb = document.getElementById('cardsTbody');
  tb.innerHTML = '';
  j.cards.forEach(c => {
    tb.innerHTML += `
      <tr>
        <td style="font-family:monospace">${c.uid}</td>
        <td>${c.name}</td>
        <td style="text-align:right">
          <button style="padding:4px 8px; border-radius:8px; border:none; background:var(--danger); color:white; cursor:pointer" onclick="deleteCard('${c.uid}')">X</button>
        </td>
      </tr>
    `;
  });
}

// Status Polling
let lastAlert = "";

async function updateStatus(){
  try{
    const r = await fetch('/status');
    const d = await r.json();
    
    document.getElementById('connDot').className = 'dot';
    document.getElementById('connText').innerText = 'Online';

    // Sensors
    document.getElementById('temp').innerText = d.temp || "--";
    document.getElementById('hum').innerText = d.hum || "--";
    
    // MQ2 Alert
    const mq2 = Number(d.mq2||0);
    const mq2Box = document.getElementById('mq2Box');
    const mq2Val = document.getElementById('mq2');
    if(mq2 > 2000){
        mq2Val.innerText = "DANGER!";
        mq2Val.className = "s-val danger-text";
        mq2Box.style.border = "1px solid var(--danger)";
    } else {
        mq2Val.innerText = "Safe";
        mq2Val.className = "s-val";
        mq2Box.style.border = "1px solid rgba(255,255,255,0.05)";
    }

    // LDR Logic updated to use config input
    let ldrVal = Number(d.ldr || 0);
    let ldrThresh = Number(document.getElementById('cfgLdr').value) || 2000;
    // Show value and state (Dark/Bright)
    let stateStr = ldrVal > ldrThresh ? "Dark" : "Bright";
    document.getElementById('ldr').innerText = `${ldrVal} (${stateStr})`;
    
    document.getElementById('rainAO').innerText = Number(d.rain_ao||4095) < 3000 ? "Raining" : "Dry";
    document.getElementById('pir').innerText = d.pir ? "Motion" : "Clear";

    // Devices
    document.getElementById('doorDesc').innerText = d.servo > 0 ? "Open" : "Locked";
    document.getElementById('ledDesc').innerText = d.led ? "On" : "Off";
    document.getElementById('fanDesc').innerText = d.relay ? "Running" : "Stopped";
    document.getElementById('ledToggle').checked = d.led;
    document.getElementById('fanToggle').checked = d.relay;

    // Face & Camera
    const camOn = d.camera_on;
    document.getElementById('camChip').innerText = camOn ? "LIVE" : "OFF";
    document.getElementById('camChip').style.background = camOn ? "var(--danger)" : "rgba(255,255,255,0.1)";
    document.getElementById('faceName').innerText = d.face_name;
    document.getElementById('faceProb').innerText = "Prob: " + Number(d.face_prob||0).toFixed(2);

    // RFID Autofill
    lastUID = (d.rfid || "NONE").toUpperCase();
    document.getElementById('rfid').innerText = lastUID;
    const chip = document.getElementById('rfidChip');
    
    if(d.rfid_valid === 1){
      chip.innerText = "Authorized"; chip.style.background = "var(--success)";
      document.getElementById('rfidTitle').innerText = "Valid Card";
    } else if (lastUID !== "NONE") {
      chip.innerText = "Denied"; chip.style.background = "var(--danger)";
      document.getElementById('rfidTitle').innerText = "Unknown Card";
    } else {
      chip.innerText = "Idle"; chip.style.background = "rgba(255,255,255,0.1)";
      document.getElementById('rfidTitle').innerText = "RFID Scanner";
    }

    if(currentTab === 'settings' && lastUID !== "NONE"){
        const inp = document.getElementById('cardUID');
        if(inp.value !== lastUID){
            inp.value = lastUID;
            showToast("Scanned UID: " + lastUID);
        }
    }

    // Face Reg UI
    if(d.is_capturing){
        document.getElementById('regProgressBox').style.display = 'block';
        const pct = (d.capture_progress / d.capture_target) * 100;
        document.getElementById('regFill').style.width = pct + "%";
        document.getElementById('regCount').innerText = d.capture_progress + "/" + d.capture_target;
        document.getElementById('regStatus').innerText = "Capturing...";
    } else if (d.is_training) {
        document.getElementById('regProgressBox').style.display = 'block';
        document.getElementById('regFill').style.width = "100%";
        document.getElementById('regStatus').innerText = "Training Model... Please Wait";
        document.getElementById('regStatus').style.color = "var(--warning)";
    } else {
        if(document.getElementById('regStatus').innerText.includes("Training")){
             document.getElementById('regStatus').innerText = "Done!";
             document.getElementById('regStatus').style.color = "var(--success)";
             setTimeout(()=> {document.getElementById('regProgressBox').style.display = 'none';}, 3000);
        }
    }

    // Toast Alert
    if(d.alert && d.alert !== lastAlert){
      lastAlert = d.alert;
      showToast(d.alert);
    }

  } catch(e){
    document.getElementById('connDot').className = 'dot offline';
    document.getElementById('connText').innerText = 'Offline';
  }
}
setInterval(updateStatus, 1000);
updateStatus();
</script>
</body>
</html>
'''

# ================== FLASK ROUTES ==================
@app.route("/")
def index(): return render_template_string(HTML_PAGE, logo=logo_base64)

@app.route("/status")
def get_status():
    with state_lock: return jsonify(dict(state))

@app.route("/video_feed")
def video_feed(): return Response(gen_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/config/get")
def api_config_get():
    with config_lock: return jsonify(sys_config)

@app.route("/api/config/update", methods=["POST"])
def api_config_update():
    global cam_restart_flag, ser
    data = request.get_json(silent=True) or {}
    resp_msg = "Saved"
    
    with config_lock:
        old_port = sys_config["serial_port"]
        old_cam = sys_config["cam_src"]
        
        if "serial_port" in data: sys_config["serial_port"] = data["serial_port"]
        if "cam_src" in data: sys_config["cam_src"] = data["cam_src"]
        for k in ["mq2_thresh", "ldr_thresh", "rain_thresh"]:
            if k in data: sys_config[k] = int(data[k])
            
        if (sys_config["serial_port"] != old_port) or (ser is None) or (not ser.is_open):
            ok, msg = reconnect_serial(sys_config["serial_port"], sys_config["baudrate"])
            resp_msg = msg if ok else f"Error: {msg}"
            
        if str(sys_config["cam_src"]) != str(old_cam): 
            cam_restart_flag = True
    
    return jsonify({"ok": True, "message": resp_msg})

@app.route("/api/lcd/send", methods=["POST"])
def api_lcd_send():
    msg = (request.get_json() or {}).get("message", "")
    send_lcd_update(force_msg=msg) # Update state & send full package
    return jsonify({"ok": True})

@app.route("/api/door/<action>", methods=["POST"])
def api_door(action):
    if action == "open": open_door_logic("Manual Web")
    else:
        send_cmd("SERVO_CLOSE")
        send_lcd_update("Door Closed")
        with state_lock: state["servo"] = 0
    return jsonify({"ok": True})

@app.route("/api/<dev>/<action>", methods=["POST"])
def api_dev(dev, action):
    # REMOVED MANUAL TIMERS
    is_on = (action == "on")
    if dev == "led":
        send_cmd("LED_ON" if is_on else "LED_OFF")
        with state_lock: state["led"] = 1 if is_on else 0
    elif dev == "fan":
        send_cmd("RELAY_ON" if is_on else "RELAY_OFF")
        with state_lock: state["relay"] = 1 if is_on else 0
    elif dev == "cam":
        if action == "start": start_face()
        else: stop_face()
    return jsonify({"ok": True})

@app.route("/api/face/register", methods=["POST"])
def api_face_reg():
    name = (request.get_json() or {}).get("name", "").strip()
    if not name: return jsonify({"ok": False}), 400
    with state_lock:
        if state["camera_on"] == 0: return jsonify({"ok": False, "message":"Start Cam First"}), 400
        state["is_capturing"] = True
        state["capture_name"] = name
        state["capture_progress"] = 0
    return jsonify({"ok": True})

@app.route("/api/rfid/list")
def api_rfid_list():
    with cards_lock: cards = [{"uid":k, "name":v} for k,v in allowed_cards.items()]
    return jsonify({"cards": cards})

@app.route("/api/rfid/register", methods=["POST"])
def api_rfid_reg():
    d = request.get_json() or {}
    if save_card(d.get("uid"), d.get("name")): return jsonify({"ok":True})
    return jsonify({"ok":False}), 500

@app.route("/api/rfid/delete/<uid>", methods=["POST"])
def api_rfid_del(uid):
    delete_card(uid)
    return jsonify({"ok":True})

# ================== MAIN ==================
if __name__ == "__main__":
    threading.Thread(target=serial_writer, daemon=True).start()
    threading.Thread(target=serial_reader, daemon=True).start()
    with state_lock: state["camera_on"] = 0
    app.run(host="0.0.0.0", port=5000, debug=False)