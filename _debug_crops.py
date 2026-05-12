import cv2, os, pickle, glob
import numpy as np
from ultralytics import YOLO

def extract_features_rich(crop):
    r = cv2.resize(crop, (128, 128))
    hsv_raw = cv2.cvtColor(r, cv2.COLOR_BGR2HSV)
    h_raw, s_raw, v_raw = cv2.split(hsv_raw)
    uv_mean_brightness = float(np.mean(v_raw))
    uv_max_brightness  = float(np.max(v_raw))
    uv_bright_ratio    = float(np.mean(v_raw > 180))
    uv_sat_spike_ratio = float(np.mean(s_raw > 150))
    uv_mask     = cv2.inRange(hsv_raw, (100, 30, 30), (160, 255, 255))
    uv_hue_hist = cv2.calcHist([hsv_raw], [0], uv_mask, [60], [100, 160])
    cv2.normalize(uv_hue_hist, uv_hue_hist)
    uv_hue_flat = uv_hue_hist.flatten()
    lab = cv2.cvtColor(r, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    limg  = cv2.merge((clahe.apply(l), a, b))
    norm  = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    hsv = cv2.cvtColor(norm, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    hist_hs = cv2.calcHist([hsv], [0, 1], None, [64, 32], [0, 180, 0, 256])
    cv2.normalize(hist_hs, hist_hs)
    l_ch, a_ch, b_ch = cv2.split(cv2.cvtColor(norm, cv2.COLOR_BGR2LAB))
    def chan_stats(c):
        flat = c.flatten().astype(np.float32)
        return [float(np.mean(flat)), float(np.std(flat)), float(np.percentile(flat, 25)), float(np.percentile(flat, 75))]
    stats = (chan_stats(h_ch) + chan_stats(s_ch) + chan_stats(v_ch) + chan_stats(l_ch) + chan_stats(a_ch) + chan_stats(b_ch))
    gray    = cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return np.concatenate([[uv_mean_brightness, uv_max_brightness, uv_bright_ratio, uv_sat_spike_ratio], uv_hue_flat, hist_hs.flatten(), stats, [lap_var]])

# Load classifier
with open("gb_classifier.pkl", "rb") as f:
    clf = pickle.load(f)

for fname in ['20260417_143607_nocar.jpg', '20260417_151305_nocar.jpg']:
    path = f"new_pipeline/crops/oil_leak/{fname}"
    if os.path.exists(path):
        c = cv2.imread(path)
        f = extract_features_rich(c)
        prob = clf.predict_proba([f])[0]
        print(f"ORIGINAL TRAIN CROP {fname}: shape={c.shape}, Leak Prob = {prob[1]:.4f}")
    
    path_test = f"test/crops/{fname}"
    if os.path.exists(path_test):
        c = cv2.imread(path_test)
        f = extract_features_rich(c)
        prob = clf.predict_proba([f])[0]
        print(f"TEST INFERENCE CROP {fname}: shape={c.shape}, Leak Prob = {prob[1]:.4f}")
