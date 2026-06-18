# 🅿️ Parking Space Occupancy Detector

A YOLOv8-powered web application for real-time detection and classification of parking space occupancy using Streamlit.

## Features

- 🚗 Real-time parking space detection
- 📊 Occupancy rate analysis
- 🎨 Color-coded visualization (Green: Free, Red: Occupied, Yellow: Partial)
- 📥 Easy image upload interface
- 📋 Detailed detection metrics and statistics
- 💾 Download annotated results

## Prerequisites

- Python 3.9+
- pip package manager
- Model file: `models/best.pt` (6 MB)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/misbahkhattak718-pixel/parking-space-detector.git
cd parking-space-detector
```

### 2. Install Git LFS (for model file)

The model file is tracked with Git LFS. Install it:

**Windows:**
```bash
winget install GitHub.GitLFS
git lfs install
```

**macOS:**
```bash
brew install git-lfs
git lfs install
```

**Linux:**
```bash
sudo apt-get install git-lfs
git lfs install
```

### 3. Pull model file

```bash
git lfs pull
```

### 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 5. Verify setup

```bash
python setup.py
```

Expected output:
```
✅ Model found at .../models/best.pt
   File size: 5.96 MB
```

## Usage

### Running the Streamlit App Locally

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

### Features

1. **Upload Image**: Upload a parking lot image (JPG/PNG)
2. **Adjust Settings**:
   - Confidence Threshold: Filter detections by confidence score
   - NMS IoU Threshold: Non-Maximum Suppression overlap threshold
   - Show Confidence Scores: Toggle confidence display on boxes
3. **Run Detection**: Click "🔍 Detect" to analyze the image
4. **View Results**:
   - Side-by-side original and annotated images
   - Occupancy metrics and statistics
   - Detailed detection table
5. **Download**: Save the annotated image

## Project Structure

```
parking-space-detector/
├── app.py                    # Streamlit application
├── setup.py                  # Setup verification script
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .gitattributes            # Git LFS configuration
├── .gitignore                # Git ignore rules
├── .streamlit/
│   └── config.toml          # Streamlit configuration
├── models/
│   └── best.pt              # YOLOv8 model weights (Git LFS)
└── utils/
    ├── __init__.py
    └── detector.py
```

## Model Information

- **Architecture**: YOLOv8 Nano
- **File Size**: 5.96 MB
- **Inference Time**: < 1 second per image
- **Classes**: 
  - 0 - Free (empty parking space)
  - 1 - Occupied (space with vehicle)
  - 2 - Partially Occupied (space partially occupied)

## Requirements

- **streamlit**: Web app framework
- **ultralytics**: YOLOv8 implementation
- **opencv-python-headless**: Image processing (headless for servers)
- **numpy**: Numerical computing
- **pandas**: Data analysis
- **Pillow**: Image handling
- **PyYAML**: Configuration
- **torch**: Deep learning framework
- **torchvision**: Vision utilities

## Deployment

### Streamlit Cloud (Recommended)

**Important: Model File Handling**

The model file (`models/best.pt`) is 6 MB and needs to be handled carefully for deployment.

#### Option A: Using Git LFS (Recommended)

1. **Install Git LFS:**
   ```bash
   # Windows
   winget install GitHub.GitLFS
   
   # macOS
   brew install git-lfs
   
   # Linux
   sudo apt-get install git-lfs
   ```

2. **Initialize Git LFS:**
   ```bash
   cd parking-space-detector
   git lfs install
   git lfs pull
   ```

3. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Add model with Git LFS"
   git push origin main
   ```

4. **Deploy on Streamlit Cloud:**
   - Go to [Streamlit Cloud](https://streamlit.io/cloud)
   - Click "New app"
   - Select your repository
   - Set "App URL" to `app.py`
   - Click Deploy!

#### Option B: Manual Model Upload

If Git LFS doesn't work on Streamlit Cloud:

1. **Create a release on GitHub:**
   - Go to Releases
   - Create new release `v1.0`
   - Upload `models/best.pt`
   - Publish

2. **Update `download_model.py`:**
   ```python
   github_url = "https://github.com/YOUR-USERNAME/parking-space-detector/releases/download/v1.0/best.pt"
   ```

3. **Add to deployment:**
   - Create `streamlit_app.py` (already provided)
   - Streamlit Cloud will run `streamlit_app.py` as entry point

#### Option C: No Model (Development Only)

The app will show helpful error messages if model is missing:

```
❌ Model Not Found

Train the model:
python convert_dataset.py
python train.py
```

### Verify Deployment

After deploying, you can:
1. Test with sample parking lot images
2. Check occupancy detection works
3. Download annotated results

### Other Platforms

Can be deployed on:
- Heroku (requires Procfile)
- AWS (Lambda + API Gateway)
- Azure (Container Instances)
- Google Cloud (Cloud Run)
- Railway
- Render

## Training (Optional)

To train a new model on your own dataset:

```bash
python convert_dataset.py
python train.py
```

Model will be saved to `models/best.pt`

## Troubleshooting

### ❌ "Model not found" error

**Solution**: Ensure Git LFS is installed and model is pulled:
```bash
git lfs install
git lfs pull
```

### ❌ "cv2 import error" on Streamlit Cloud

**Fixed**: Using `opencv-python-headless` and lazy imports

### ❌ Model loads slowly

**Expected**: First load caches the model (~5-10 seconds). Subsequent runs are instant.

## Performance

- **Model**: YOLOv8 Nano (Fast, 6 MB)
- **Input Size**: Adaptive (max 1920px)
- **Inference Speed**: ~500-800ms per image
- **Memory Usage**: ~200-300 MB

## Author

**Misbah Khattak**
- GitHub: [@misbahkhattak718-pixel](https://github.com/misbahkhattak718-pixel)
- Email: misbahkhattak718@gmail.com

## License

This project is open source and available under the MIT License.

## Support

For issues, feature requests, or questions, please:
1. Check existing [GitHub Issues](https://github.com/misbahkhattak718-pixel/parking-space-detector/issues)
2. Create a new issue with detailed description
3. Include screenshots if applicable

## Acknowledgments

- Built with [Streamlit](https://streamlit.io)
- Model trained with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- OpenCV for image processing

---

**Happy Parking! 🚗✨**
