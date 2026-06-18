"""
Setup script for Parking Space Detector

This script ensures the model is available for the Streamlit app.
Run this before deploying to ensure models/best.pt exists.
"""

import os
from pathlib import Path

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "best.pt"

def ensure_model_exists():
    """Ensure model file exists."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    if MODEL_PATH.exists():
        print(f"✅ Model found at {MODEL_PATH}")
        print(f"   File size: {MODEL_PATH.stat().st_size / 1024 / 1024:.2f} MB")
        return True
    
    print("❌ Model not found!")
    print(f"\nTo create the model, run:")
    print("  python convert_dataset.py")
    print("  python train.py")
    print(f"\nModel will be saved to: {MODEL_PATH}")
    return False

if __name__ == "__main__":
    print("🅿️  Parking Space Detector - Setup")
    print("=" * 50)
    ensure_model_exists()
