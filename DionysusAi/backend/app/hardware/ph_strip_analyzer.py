import cv2
import numpy as np

# --- 1. CALIBRATION DATA ---

PH_SCALE = [
    {"val": 3.0, "bgr": [45, 45, 200]},
    {"val": 3.5, "bgr": [60, 130, 240]},
    {"val": 4.0, "bgr": [80, 210, 240]}
]

SULFATE_MATRIX = [
    [[211, 122, 54], [212, 144, 40], [221, 146, 46], [205, 137, 53]], 
    [[214, 117, 70], [215, 125, 60], [225, 141, 51], [213, 133, 55]], 
    [[220, 124, 81], [219, 126, 84], [227, 134, 72], [218, 138, 58]],
    [[228, 132, 91], [225, 132, 90], [223, 127, 87], [207, 120, 65]] 
]
SULFATE_VALUES = [200, 400, 800, 1200]

# --- 2. LOGIC ENGINES ---

def bgr_to_lab(bgr):
    pixel = np.uint8([[bgr]])
    return cv2.cvtColor(pixel, cv2.COLOR_BGR2Lab)[0][0].astype(float)

def calculate_ph(sampled_bgr):
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

def calculate_sulfates(frame, rois):
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
    
    return round(np.mean(pad_results) / 1000.0, 3) 

# --- 3. MAIN APPLICATION ---

phone_url = "http://192.168.1.21:8080/video" 
cap = cv2.VideoCapture(phone_url)
current_mode = "pH"

ph_box = (250, 200, 60, 60)
sulfate_rois = [(250, 120, 40, 30), (250, 170, 40, 30), (250, 220, 40, 30), (250, 270, 40, 30)]
white_box = (400, 200, 50, 50)

print("CONSOLE: 'M' to toggle mode, 'S' to scan, 'Q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break

    cv2.rectangle(frame, (white_box[0], white_box[1]), (white_box[0]+white_box[2], white_box[1]+white_box[3]), (255,255,255), 2)

    if current_mode == "pH":
        cv2.rectangle(frame, (ph_box[0], ph_box[1]), (ph_box[0]+ph_box[2], ph_box[1]+ph_box[3]), (0, 0, 255), 2)
    else:
        for (x, y, w, h) in sulfate_rois:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)

    cv2.putText(frame, f"MODE: {current_mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.imshow('Wine Feature Extractor', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('m'):
        current_mode = "SULPHATES" if current_mode == "pH" else "pH"
    
    elif key == ord('s'):
        white_roi = frame[white_box[1]:white_box[1]+white_box[3], white_box[0]:white_box[0]+white_box[2]]
        avg_white = np.mean(white_roi, axis=(0, 1))
        correction = 255.0 / (avg_white + 1e-6)
        corrected_frame = np.clip(frame * correction, 0, 255).astype(np.uint8)

        if current_mode == "pH":
            roi = corrected_frame[ph_box[1]:ph_box[1]+ph_box[3], ph_box[0]:ph_box[0]+ph_box[2]]
            val = calculate_ph(np.mean(roi, axis=(0,1)))
            print(f"RESULT | pH: {val}")
        else:
            val = calculate_sulfates(corrected_frame, sulfate_rois)
            print(f"RESULT | Sulphates: {val} g/L")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()