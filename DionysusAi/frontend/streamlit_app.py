import streamlit as st
import requests
import cv2
import serial
import numpy as np
import time
import json

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="DionysusAI 🍷",
    page_icon="🍷",
    layout="wide"
)

st.markdown("""
    <style>
    .wine-card {
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #800020;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🍷 DionysusAI – Intelligent Wine Companion")

# ---------------- Sidebar ----------------
page = st.sidebar.radio(
    "Navigate",
    [
        "Wine Quality Prediction",
        "Food Pairings",
        "Wine Recommendations",
        "Wine Mentor Chat",
    ]
)

# --- 1. HARDWARE CONFIGURATION ---
MQ3_PORT = 'COM4'
CAMERA_URL = "http://192.168.1.21:8080/video"

# --- 2. CALIBRATION DATA ---
PH_SCALE = [
    {"val": 3.0, "bgr": [45, 45, 200]},
    {"val": 3.5, "bgr": [60, 130, 240]},
    {"val": 4.0, "bgr": [80, 210, 240]}
]

SULFATE_MATRIX = [
    [[211, 122, 54], [212, 144, 40], [221, 146, 46], [205, 137, 53]], # Pad 1
    [[214, 117, 70], [215, 125, 60], [225, 141, 51], [213, 133, 55]], # Pad 2
    [[220, 124, 81], [219, 126, 84], [227, 134, 72], [218, 138, 58]], # Pad 3
    [[228, 132, 91], [225, 132, 90], [223, 127, 87], [207, 120, 65]]  # Pad 4
]

SULFATE_VALUES = [200, 400, 800, 1200]


PH_BOX = (250, 200, 60, 60)
WHITE_BOX = (400, 200, 50, 50)
SULFATE_ROIS = [
    (250, 120, 40, 30),
    (250, 170, 40, 30),
    (250, 220, 40, 30),
    (250, 270, 40, 30)
]

# --- 3. HELPER LOGIC (YOUR MATH) ---
def bgr_to_lab(bgr):
    pixel = np.uint8([[bgr]])
    return cv2.cvtColor(pixel, cv2.COLOR_BGR2Lab)[0][0].astype(float)

def calculate_ph_logic(sampled_bgr):
    target_lab = bgr_to_lab(sampled_bgr)
    distances = []
    for item in PH_SCALE:
        ref_lab = bgr_to_lab(item["bgr"])
        dist = np.linalg.norm(target_lab - ref_lab)
        distances.append((dist, item["val"]))
    distances.sort(key=lambda x: x[0])
    d1, v1 = distances[0]
    d2, v2 = distances[1]
    weight = d1 / (d1 + d2 + 1e-6)
    return round(v1 + weight * (v2 - v1), 2)

def calculate_sulfates_logic(frame, rois):
    pad_results = []
    for i, (x, y, w, h) in enumerate(rois):
        roi = frame[y:y+h, x:x+w]
        avg_bgr = np.mean(roi, axis=(0,1))
        avg_rgb = avg_bgr[::-1] 

        dists = [np.linalg.norm(np.array(avg_rgb) - np.array(ref)) for ref in SULFATE_MATRIX[i]]
        idx_sorted = np.argsort(dists)
        i1, i2 = idx_sorted[0], idx_sorted[1]
        
        weight = dists[i1] / (dists[i1] + dists[i2] + 1e-6)
        val = SULFATE_VALUES[i1] + weight * (SULFATE_VALUES[i2] - SULFATE_VALUES[i1])
        pad_results.append(val)
    
    return round(np.mean(pad_results) / 1000.0, 3) # g/L

def apply_white_balance(frame):
    """Applies the correction based on the WHITE_BOX area"""
    x, y, w, h = WHITE_BOX
    white_roi = frame[y:y+h, x:x+w]
    avg_white = np.mean(white_roi, axis=(0, 1))
    correction = 255.0 / (avg_white + 1e-6)
    return np.clip(frame * correction, 0, 255).astype(np.uint8)

# -------- Helper --------
def predict_quality(wine_data):
    r = requests.post(f"{API_BASE}/api/predict", json=wine_data)
    return r.json()

# --- Chemical sensors ---
def process_camera(mode="pH", preview_only=False):
    status = st.empty()
    try:
        cap = cv2.VideoCapture(CAMERA_URL)

        if not preview_only:
            time.sleep(1.0)
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            status.error("❌ Failed to connect to IP Webcam.")
            return None
        if preview_only:
            cv2.rectangle(frame, (WHITE_BOX[0], WHITE_BOX[1]),
                          (WHITE_BOX[0]+WHITE_BOX[2], WHITE_BOX[1]+WHITE_BOX[3]), (255,255,255), 2)
            
            if mode == "pH":
                cv2.rectangle(frame, (PH_BOX[0], PH_BOX[1]),
                              (PH_BOX[0]+PH_BOX[2], PH_BOX[1]+PH_BOX[3]), (0,0,255), 2)
            else:
                for (x, y, w, h) in SULFATE_ROIS:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            
            return frame 
        # --- ANALYSIS MODE---
        # 1. White Balance
        corrected_frame = apply_white_balance(frame)
        
        # 2. Extract & Calculate
        if mode == "pH":
            x, y, w, h = PH_BOX
            roi = corrected_frame[y:y+h, x:x+w]
            val = calculate_ph_logic(np.mean(roi, axis=(0,1)))
            return val
            
        elif mode == "Sulphates":
            val = calculate_sulfates_logic(corrected_frame, SULFATE_ROIS)
            return val
    except Exception as e:
        status.error(f"Camera Error: {e}")
        return None

# --- Alcphol Sensor function ---
def scan_alcohol_hardware():
    """Connects to COM4, reads MQ-3 sensor, returns ABV float"""
    status_box = st.empty()
    progress = st.progress(0)
    
    try:
        # 1. Connect
        status_box.info("🔌 Connecting to Sensor on COM4...")
        ser = serial.Serial('COM4', 115200, timeout=1)
        time.sleep(2) 
        
        # 2. Baseline (Clean Air) - 10 seconds
        status_box.info("💨 Reading Baseline (Keep sensor in clean air)...")
        baselines = []
        for i in range(20):
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if 'VOLTAGE:' in line:
                try:
                    v = float(line.split(':')[1])
                    baselines.append(v)
                except: pass
            progress.progress((i + 1) / 30) 
            time.sleep(1)
            
        avg_baseline = np.mean(baselines) if baselines else 1.0
        # 3. Exposure (Wine) - 20 seconds
        status_box.warning("🍷 ACTION REQUIRED: Hold sensor over wine now!")
        voltages = []
        for i in range(30):
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if 'VOLTAGE:' in line:
                try:
                    v = float(line.split(':')[1])
                    voltages.append(v)
                except: pass
            progress.progress((i + 11) / 30)
            time.sleep(1)
            
        ser.close()
        status_box.success("✅ Scan Complete!")
        time.sleep(1)
        status_box.empty()
        progress.empty()
        peak = max(voltages) if voltages else avg_baseline
        delta = peak - avg_baseline
        abv = -4.31 * np.log(max(delta, 1e-6)) + 15.21
        return round(abv, 2)
    except Exception as e:
        status_box.error(f"Hardware Error: {e}")
        if 'ser' in locals() and ser.is_open:
            ser.close()
        return None

# ================= PAGE 1: PREDICTION =================
if page == "Wine Quality Prediction":
    st.header("🧪 Wine Quality Prediction")
    if 'ph_val' not in st.session_state: st.session_state.ph_val = 3.21
    if 'sulph_val' not in st.session_state: st.session_state.sulph_val = 0.51
    if 'alc_val' not in st.session_state: st.session_state.alc_val = 10.3

    st.subheader("📡 Sensor Inputs")
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("**1. Alcohol Sensor (MQ-3)**")
        if st.button("🌡️ Scan Alcohol (MQ-3)"):
            val = scan_alcohol_hardware() 
            if val is not None:
                st.session_state.alc_val = val
                st.rerun()
    with c2:
        st.markdown("**2. Camera Analysis**")
        cam_mode = st.radio("Test Strip:", ["pH", "Sulphates"], horizontal=True)
        col_cam1, col_cam2 = st.columns(2)
        if col_cam1.button("👁️ Check Alignment"):
            frame = process_camera(mode=cam_mode, preview_only=True)
            if frame is not None:
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        if col_cam2.button("📸 Capture & Analyze"):
            val = process_camera(mode=cam_mode, preview_only=False)
            if val is not None:
                if cam_mode == "pH": st.session_state.ph_val = val
                else: st.session_state.sulph_val = val
                st.success(f"Updated {cam_mode}!")
                time.sleep(1); st.rerun()

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        wine_type = st.selectbox("Wine Type", ["Red", "White"])
        fixed_acidity = st.slider("Fixed Acidity", 3.0, 15.0, 7.0)
        volatile_acidity = st.slider("Volatile Acidity", 0.1, 1.5, 0.31)
        citric_acid = st.slider("Citric Acid", 0.0, 1.0, 0.31)
        residual_sugar = st.slider("Residual Sugar", 0.0, 20.0, 5.0)
        chlorides = st.slider("Chlorides", 0.01, 0.2, 0.047)
    with col2:
        free_so2 = st.slider("Free SO₂", 1.0, 80.0, 34.0)
        density = st.slider("Density", 0.990, 1.005, 0.9949)
        pH = st.slider("pH", 2.8, 4.0, value=st.session_state.ph_val)
        sulphates = st.slider("Sulphates", 0.3, 2.0, value=st.session_state.sulph_val)
        alcohol = st.slider("Alcohol %", 8.0, 15.0, value=st.session_state.alc_val)

    if st.button("Predict Quality", type="primary"):
        wine_data = {
            "type": 1 if wine_type == "White" else 0, "fixed_acidity": fixed_acidity,
            "volatile_acidity": volatile_acidity, "citric_acid": citric_acid,
            "residual_sugar": residual_sugar, "chlorides": chlorides,
            "free_sulfur_dioxide": free_so2, "density": density,
            "pH": pH, "sulphates": sulphates, "alcohol": alcohol
        }
        r = requests.post(f"{API_BASE}/api/predict", json=wine_data)
        res = r.json()
        if res.get("success"):
            st.balloons()
            st.success(f"🍷 Quality Label: **{res['quality_label']}**")
            st.metric("Score", f"{res['quality_score']:.2f}")

# ================= PAGE 2: FOOD PAIRINGS =================
if page == "Food Pairings":
    st.header("🍷 Wine & Food Pairing AI")

    c1, c2, c3 = st.columns(3)
    with c1: 
        cur_alc = st.number_input("Alcohol %", 8.0, 16.0, float(st.session_state.get('alc_val', 12.0)))
    with c2: 
        cur_ph = st.number_input("pH Level", 2.8, 4.0, float(st.session_state.get('ph_val', 3.3)))
    with c3: 
        w_type = st.selectbox("Wine Type", ["Red", "White"])

    if st.button("Generate Chef's Pairings", type="primary"):
        payload = {
            "type": 1 if w_type == "White" else 0, 
            "pH": cur_ph, 
            "alcohol": cur_alc,
            "fixed_acidity": 7.0, 
            "volatile_acidity": 0.5, 
            "citric_acid": 0.3, 
            "residual_sugar": 2.5, 
            "chlorides": 0.05, 
            "free_sulfur_dioxide": 30.0,
            "density": 0.996, 
            "sulphates": 0.6
        }

        with st.spinner("Chef Dionysus is tasting the wine..."):
            try:
                r = requests.post(f"{API_BASE}/api/pairings/recommend", json=payload, timeout=30)
                
                if r.status_code == 200:
                    data = r.json()
                    st.divider()
                    st.subheader("Analysis & Recommendations")
                    
                    col_a, col_b = st.columns([1, 2])
                    
                    with col_a:
                        st.info(f"**Body:** {data.get('body')}\n\n**Acidity:** {data.get('acidity')}")

                        temp = data.get('serving_temp', 'Room Temp')
                        if len(str(temp)) > 20: 
                            temp = "10-15°C (50-59°F)"
                        st.metric("Serving Temp", temp)

                    with col_b:
                        st.success(data.get('analysis', 'A great choice for this profile.'))

                        pairings_raw = data.get('pairings', [])
                        
                        if isinstance(pairings_raw, str):
                            try:
                                pairings_list = json.loads(pairings_raw.replace("'", '"'))
                            except:
                                pairings_list = [pairings_raw]
                        else:
                            pairings_list = pairings_raw

                        for dish in pairings_list:
                            if dish and len(str(dish)) > 1: 
                                st.markdown(f"🍴 **{dish.strip(' \"[]')}**")
                else:
                    st.error(f"The kitchen is closed (Error {r.status_code})")
                    
            except Exception as e:
                st.error(f"Connection Failed: {e}")
                
# ================= PAGE 3: RECOMMENDATIONS =================
elif page == "Wine Recommendations":
    st.header("🍷 Find Your Next Bottle")
    user_query = st.text_input("What are you looking for?", placeholder="e.g. A spicy red from Italy")
    
    if st.button("Search Database"):
        with st.spinner("Searching..."):
            r = requests.post(f"{API_BASE}/api/recommend", json={"query": user_query, "limit": 3})
            data = r.json()
            if data.get("success") and data.get("wines"):
                for wine in data['wines']:
                    with st.container():
                        st.markdown(f"""<div class='wine-card'>
                            <h3>{wine.get('title')}</h3>
                            <p><b>{wine.get('variety')}</b> | {wine.get('province')}, {wine.get('country')}</p>
                            <p><i>{wine.get('description')}</i></p>
                            <span style='color: #ffd700;'>Price: ${wine.get('price', 'N/A')}</span>
                        </div>""", unsafe_base_html=True)
            else:
                st.warning("⚠️ Recommendation Engine is currently offline or database is empty.")

# ================= PAGE 4: MENTOR CHAT =================
elif page == "Wine Mentor Chat":
    st.header("DionysusAI Mentor")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Greetings, mortal. What would you like to know about the nectar of the gods?"}]
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    if prompt := st.chat_input("Ask about tannins, regions, or etiquette..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Consulting the wine gods..."):
                try:
                    r = requests.post(
                        f"{API_BASE}/api/mentor/chat", 
                        json={"message": prompt},
                        timeout=45
                    )
                    
                    if r.status_code == 200:
                        response_text = r.json().get("response", "I'm speechless... try again.")
                        st.markdown(response_text)
                        st.session_state.messages.append({"role": "assistant", "content": response_text})
                    else:
                        st.error(f"The gods are angry (Error {r.status_code})")
                        
                except requests.exceptions.Timeout:
                    st.error("The mentor is taking too long. Ollama might be busy.")
                except Exception as e:
                    st.error(f"Connection Error: {e}")
                
