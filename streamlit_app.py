"""
Streamlit app entry point
This file is for Streamlit Cloud deployment.
It ensures dependencies are set up before running the main app.
"""

import sys
from pathlib import Path

# Try to download model if it doesn't exist
try:
    model_path = Path(__file__).parent / "models" / "best.pt"
    if not model_path.exists():
        print("📦 Model file not found. Attempting to set up...")
        print("   Note: For production, ensure models/best.pt exists")
except Exception as e:
    print(f"⚠️  Warning: {e}")

# Import and run the main app
try:
    from app import *  # noqa: F401,F403
except ImportError as e:
    print(f"❌ Error importing app: {e}")
    sys.exit(1)
