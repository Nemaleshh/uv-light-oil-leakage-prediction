import os
import cv2
import pickle

from retrain_classifier import extract_features

BASE = r"C:\Users\hnema\OneDrive\Desktop\stellatis\newengil2\newengil"
CLF_PATH = os.path.join(BASE, "app_v2", "models", "gb_classifier.pkl")
IMAGE_PATH = r"C:\Users\hnema\.gemini\antigravity\brain\e3fbb9d8-da09-4900-83b7-90260ece132f\media__1778136906703.jpg"

def main():
    print(f"Loading image: {IMAGE_PATH}")
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print("Failed to load image.")
        return

    print("Loading classifier...")
    with open(CLF_PATH, "rb") as f:
        clf = pickle.load(f)

    print("Extracting features...")
    feats = extract_features(img).reshape(1, -1)
    
    print("Predicting...")
    pred = clf.predict(feats)[0]
    proba = clf.predict_proba(feats)[0]
    conf = max(proba) * 100

    pred_label = "OIL LEAK" if pred == 1 else "NO LEAK"

    print("\n--- RESULTS ---")
    print(f"Prediction: {pred_label}")
    print(f"Confidence: {conf:.1f}%")
    print("---------------")

if __name__ == "__main__":
    main()
