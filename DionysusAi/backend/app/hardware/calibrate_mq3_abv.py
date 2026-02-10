# import serial
# import time
# import numpy as np
# from scipy.optimize import curve_fit  # 

# # ESP32 port
# ser = serial.Serial('COM4', 115200, timeout=1)
# time.sleep(3)  # Wait for boot

# def collect_data(seconds):
#     voltages = []
#     start = time.time()
#     print(f"Collecting for {seconds}s...")
#     while time.time() - start < seconds:
#         line = ser.readline().decode('utf-8').strip()
#         if 'VOLTAGE:' in line:
#             try:
#                 v = float(line.split(':')[1])
#                 voltages.append(v)
#                 print(f"{v:.3f}V")
#             except:
#                 pass
#         time.sleep(0.1)
#     return voltages

# # Calibration function (log: ABV = a * ln(delta_v) + b)
# def abv_model(delta_v, a, b):
#     return a * np.log(delta_v + 1e-6) + b  # Avoid log(0)

# # Wines info - change ABV to your exact labels
# wines = [
#     {"name": "White 11%", "abv": 11.0},
#     {"name": "White 12.5%", "abv": 12.5},
#     {"name": "Red 14%", "abv": 14.0}
# ]

# deltas = []
# for wine in wines:
#     input(f"Prepare {wine['name']} ({wine['abv']}%). Press Enter for 20s baseline...")
#     baseline_data = collect_data(20)
#     baseline = np.mean(baseline_data) if baseline_data else 1.0
#     input("Now expose to headspace (5cm, 30s). Press Enter...")
#     exposure_data = collect_data(30)
#     peak = np.max(exposure_data) if exposure_data else baseline
#     delta = peak - baseline
#     deltas.append(delta)
#     print(f"Delta for {wine['name']}: {delta:.3f}V (baseline {baseline:.3f}V, peak {peak:.3f}V)")

# # Fit curve
# known_abvs = [w['abv'] for w in wines]
# popt, _ = curve_fit(abv_model, deltas, known_abvs, p0=[3, 8])  # Starting guess
# a, b = popt
# print(f"\nCalibration: a={a:.2f}, b={b:.2f}")
# with open('mq3_calib.txt', 'w') as f:
#     f.write(f"a={a}\nb={b}")

# # Future estimate section
# while True:
#     if input("Test a random wine? (y/n): ").lower() != 'y':
#         break
#     input("Press Enter for 20s baseline...")
#     baseline_data = collect_data(20)
#     baseline = np.mean(baseline_data)
#     input("Expose to headspace (30s)...")
#     exposure_data = collect_data(30)
#     peak = np.max(exposure_data)
#     delta = peak - baseline
#     abv_est = a * np.log(delta + 1e-6) + b
#     print(f"Estimated ABV: {abv_est:.1f}% (delta {delta:.3f}V)")

# ser.close()

import serial
import time
import numpy as np
import sys

# --- 1. CONFIGURATION & CALIBRATION ---
PORT = 'COM4'
BAUD = 115200

A_COEFF = -4.31
B_COEFF = 15.21

def get_abv(ser, is_api_mode=False):
    """
    Core logic to measure alcohol. 
    If is_api_mode is True, it removes all interactive inputs.
    """
    voltages = []
    try:
        print("STATUS: Collecting Baseline (clean air) 30s...")
        for i in range(30):
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if 'VOLTAGE:' in line:
                try:
                    v = float(line.split(':')[1])
                    voltages.append(v)
                except (ValueError, IndexError):
                    pass
            time.sleep(1)
            if not is_api_mode and i % 5 == 0:
                print(f"  ... {30-i}s remaining")

        baseline = np.mean(voltages) if voltages else 1.0

        if not is_api_mode:
            input("\nACTION REQUIRED: Expose sensor to wine headspace (5cm). Press Enter when in position...")
        else:
            print("STATUS: API Mode - Beginning 30s Exposure automatically...")

        voltages = []
        for i in range(30):
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if 'VOLTAGE:' in line:
                try:
                    v = float(line.split(':')[1])
                    voltages.append(v)
                except (ValueError, IndexError):
                    pass
            time.sleep(1)
            if not is_api_mode and i % 5 == 0:
                print(f"  ... {30-i}s remaining")

        peak = max(voltages) if voltages else baseline
        delta = peak - baseline

        abv = A_COEFF * np.log(max(delta, 1e-6)) + B_COEFF

        print(f"\n--- MEASUREMENT COMPLETE ---")
        print(f"Baseline: {baseline:.3f}V | Peak: {peak:.3f}V | Delta: {delta:.3f}V")

        print(f"RESULT_ABV: {round(abv, 2)}")
        
        return abv
    except KeyboardInterrupt:
        print("STATUS: Measurement interrupted by user/system.")
        return 0.0 
        
    

def run_manual_mode():
    """Traditional interactive loop for manual testing."""
    print(f"--- DionysusAI MQ-3 Manual Interface (Port: {PORT}) ---")
    try:
        ser = serial.Serial(PORT, BAUD, timeout=1)
        time.sleep(3) # Wait for ESP32 boot
        
        while True:
            choice = input("\nTest new wine? (y/n): ").lower()
            if choice != 'y':
                break
            get_abv(ser, is_api_mode=False)
            
        ser.close()
        print("Serial connection closed. Goodbye!")
    except Exception as e:
        print(f"FATAL ERROR: {e}")

def run_api_mode():
    """One-shot execution for FastAPI subprocess."""
    try:
        # Open, measure, print, and close immediately
        ser = serial.Serial(PORT, BAUD, timeout=1)
        time.sleep(3)
        get_abv(ser, is_api_mode=True)
        ser.close()
    except Exception as e:
        print(f"API_ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Check if we were called with 'python calibrate_mq3_abv.py --api'
    if "--api" in sys.argv:
        run_api_mode()
    else:
        run_manual_mode()