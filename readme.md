# AuthentiVox - AI-Powered Deepfake Audio Detection

Advanced Hybrid AI System for Detecting Synthetic Voice Manipulation

## Overview

**AuthentiVox** detects AI-generated voice and deepfake audio with **95%+ accuracy** using a hybrid Machine Learning and Deep Learning approach.

### Features

- Real-time audio analysis (< 5 seconds)
- Supports WAV, MP3, FLAC, OGG formats
- Confidence scoring with detailed reports
- Visual spectrogram generation
- Browser-based history tracking
- Dark mode support
- Responsive design

### How It Works

1. **Machine Learning**: Random Forest analyzes MFCC and spectral features
2. **Deep Learning**: CNN examines mel spectrogram patterns
3. **Ensemble**: Combines both models (ML: 40%, DL: 60%)

---

## Dataset

### ASVspoof 2019 Dataset

This project uses the **ASVspoof 2019** (Automatic Speaker Verification Spoofing and Countermeasures Challenge) dataset for training and evaluation.

**About the Dataset:**
- **Source**: ASVspoof Challenge organized by consortium of international research institutions
- **Purpose**: Benchmark dataset for voice anti-spoofing research
- **Content**: Real and synthetic speech samples
- **Spoofing Types**: Multiple Text-to-Speech (TTS) and Voice Conversion (VC) algorithms
- **Size**: 
  - Training: ~25,000 audio files
  - Development: ~24,000 audio files
  - Evaluation: ~71,000 audio files
- **Format**: 16 kHz, 16-bit PCM WAV files
- **License**: Free for research purposes

**Spoofing Algorithms Included:**
- A01-A06: Various TTS systems
- A07-A19: Various Voice Conversion systems

**Dataset Link**: [ASVspoof 2019 Database](https://datashare.ed.ac.uk/handle/10283/3336)

**Citation:**
```
ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech
Massimiliano Todisco, Xin Wang, Ville Vesikari, et al.
Computer Speech & Language, 2020
```

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Sumit07-git/AuthentiVox-AI.git
cd AUDIO

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run application
python app.py
```

Visit: `http://localhost:5000`

---

## Technology Stack

**Backend:**
- Python 3.8+
- Flask 2.0+
- TensorFlow 2.0+ / Keras
- Scikit-learn 1.0+
- Librosa 0.9+

**Frontend:**
- HTML5, CSS3, JavaScript ES6+
- Font Awesome 6

**AI Models:**
- Random Forest (ML)
- CNN (DL)

---

## Project Structure

```
AUDIO/
├── data/train/                 # Training data (ASVspoof 2019)
├── models/
│   ├── dl_model/              # Deep Learning model
│   └── ml_model/              # Machine Learning model
├── static/
│   ├── css/                   # Stylesheets
│   ├── js/
│   │   ├── history.js         # History page
│   │   ├── main.js            # Main JS
│   │   └── upload.js          # Upload page
│   └── uploads/               # Temp uploads
├── templates/
│   ├── base.html              # Base template
│   ├── history.html           # History page
│   ├── index.html              # Home page
│   └── upload.html            # Upload page
├── utils/
│   ├── feature_extractor.py   # Feature extraction
│   ├── hybrid_predictor.py    # Ensemble prediction
│   └── spectrogram_generator.py
├── app.py                      # Flask app
├── requirements.txt            # Dependencies
├── train_dl_model.py          # Train DL model
└── train_ml_model.py          # Train ML model
```

---

## Usage

### Web Interface

1. Open `http://localhost:5000`
2. Click "Start Detection"
3. Upload audio file (drag & drop or browse)
4. Click "Analyze Audio"
5. View results

### API

```python
import requests

url = 'http://localhost:5000/api/upload'
files = {'audio_file': open('sample.wav', 'rb')}

response = requests.post(url, files=files)
result = response.json()

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence_score']}%")
```

---

## API Endpoints

### POST /api/upload

Upload and analyze audio file.

**Response:**
```json
{
  "success": true,
  "is_fake": false,
  "confidence_score": 95,
  "prediction": "Real",
  "ml_confidence": 93.2,
  "dl_confidence": 96.8,
  "processing_time": 2.3
}
```

### GET /api/health

Health check.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true
}
```

---

## Model Training

### Train ML Model

```bash
python train_ml_model.py
```

Trains Random Forest on ASVspoof 2019 data in `data/train/` and saves to `models/ml_model/`.

### Train DL Model

```bash
python train_dl_model.py
```

Trains CNN on spectrograms from ASVspoof 2019 and saves to `models/dl_model/`.

### Dataset Structure

```
data/train/
├── real/              # Genuine speech samples
│   ├── LA_T_1000137.wav
│   ├── LA_T_1000265.wav
│   └── ...
└── fake/              # Spoofed speech samples
    ├── LA_T_1000003.wav
    ├── LA_T_1000068.wav
    └── ...
```

---

## Performance

Evaluated on ASVspoof 2019 evaluation set:

| Metric | ML Model | DL Model | Ensemble |
|--------|----------|----------|----------|
| Accuracy | 94.2% | 95.8% | **96.1%** |
| Precision | 93.8% | 95.4% | **95.9%** |
| Recall | 94.6% | 96.2% | **96.3%** |
| F1 Score | 94.2% | 95.8% | **96.1%** |

---

## Troubleshooting

**Port in use:**
```bash
python app.py --port 5001
```

**Missing models:**
```bash
python train_ml_model.py
python train_dl_model.py
```

**Module not found:**
```bash
source venv/bin/activate
pip install -r requirements.txt
```

---


## Acknowledgments

- **ASVspoof Challenge** - For providing the benchmark dataset
- **TensorFlow & Scikit-learn teams** - For ML/DL frameworks
- **Librosa developers** - For audio processing tools
- **Flask community** - For web framework

---

## Contact

- **Email**: sumitsamal08@gmail.com
- **GitHub**: https://github.com/Sumit07-git/AuthentiVox-AI.git

---

## Citation

If you use this project or the ASVspoof 2019 dataset, please cite:

```bibtex
@article{todisco2020asvspoof,
  title={ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech},
  author={Todisco, Massimiliano and Wang, Xin and Vesikari, Ville and others},
  journal={Computer Speech \& Language},
  volume={64},
  pages={101114},
  year={2020},
  publisher={Elsevier}
}
```

---

**Made with ❤️ by the AuthentiVox Team**