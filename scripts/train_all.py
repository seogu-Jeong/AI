import subprocess
import sys
import os

def run_script(script_path, args=[]):
    cmd = [sys.executable, script_path] + args
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Error running {script_path}")
        return False
    return True

def train_all():
    print("Starting Training Pipeline...")
    
    # Use demo flag for fast training as requested
    training_steps = [
        ("training/train_lstm.py", ["--demo"]),
        ("training/train_cnn.py", ["--demo"]),
        ("training/train_transformer.py", ["--demo"]),
        ("training/train_mlp.py", ["--demo"]),
    ]
    
    success_count = 0
    for script, args in training_steps:
        if run_script(script, args):
            success_count += 1
            
    print("\n" + "="*30)
    print("Training Summary")
    print("="*30)
    
    weights = [
        "weights/lstm_best.pt",
        "weights/cnn_best.pt",
        "weights/transformer_best.pt",
        "weights/mlp_best.pt"
    ]
    
    for w in weights:
        status = "EXISTS" if os.path.exists(w) else "MISSING"
        print(f"{w:30} : {status}")
    
    print(f"\nCompleted {success_count}/{len(training_steps)} training steps.")

if __name__ == "__main__":
    train_all()
