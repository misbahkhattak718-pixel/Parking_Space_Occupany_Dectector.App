"""
Download model file for deployment

This script downloads the model from GitHub releases if it doesn't exist locally.
Run this before deployment to ensure the model is available.
"""

import os
import sys
from pathlib import Path
import urllib.request
import ssl

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "best.pt"

# Disable SSL verification for downloading (optional, remove if not needed)
ssl._create_default_https_context = ssl._create_unverified_context

def download_model():
    """Download model from GitHub releases."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    if MODEL_PATH.exists():
        size_mb = MODEL_PATH.stat().st_size / 1024 / 1024
        print(f"✅ Model already exists: {MODEL_PATH} ({size_mb:.2f} MB)")
        return True
    
    print("⏳ Model not found. Attempting to download...")
    print(f"   Target: {MODEL_PATH}")
    
    # GitHub releases URL (update this with your actual release URL)
    github_url = "https://github.com/misbahkhattak718-pixel/parking-space-detector/releases/download/v1.0/best.pt"
    
    try:
        print(f"   Downloading from: {github_url}")
        urllib.request.urlretrieve(github_url, MODEL_PATH)
        
        size_mb = MODEL_PATH.stat().st_size / 1024 / 1024
        print(f"✅ Model downloaded successfully ({size_mb:.2f} MB)")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        print(f"\nTo fix this:")
        print(f"1. Train locally: python convert_dataset.py && python train.py")
        print(f"2. Or manually place models/best.pt in the repository")
        return False

if __name__ == "__main__":
    success = download_model()
    sys.exit(0 if success else 1)
