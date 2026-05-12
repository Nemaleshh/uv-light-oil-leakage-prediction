import os
import sys
import time
import numpy as np
import cv2

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from core.pipeline import LeakPipeline

def run_benchmark(num_frames=100):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "best.pt")
    clf_path = os.path.join(base_dir, "models", "gb_classifier.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(clf_path):
        print("❌ Models not found. Cannot run FPS benchmark.")
        return

    print("Loading pipeline models...")
    pipeline = LeakPipeline(model_path, clf_path)
    
    try:
        pipeline.load()
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        return

    print(f"Pipeline loaded. Running inference on {num_frames} frames...")
    
    # Generate a dummy test frame
    # We add some noise and shapes so YOLO doesn't completely skip it instantly
    frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    
    # Warmup
    for _ in range(3):
        pipeline.run(frame)
        
    start_time = time.time()
    
    for _ in range(num_frames):
        pipeline.run(frame)
        
    end_time = time.time()
    
    total_time = end_time - start_time
    fps = num_frames / total_time
    
    print("-" * 40)
    print("🚀 PERFORMANCE BENCHMARK RESULTS")
    print("-" * 40)
    print(f"Total Frames Processed : {num_frames}")
    print(f"Total Time Elapsed     : {total_time:.2f} seconds")
    print(f"Average Inference FPS  : {fps:.2f} FPS")
    print("-" * 40)

if __name__ == "__main__":
    run_benchmark(100)
