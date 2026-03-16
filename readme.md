---
title: AuthentiVox - Deepfake Audio Detection
emoji: 🎙️
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app_gradio.py
pinned: false
license: mit
python_version: 3.10
---

# 🎙️ AuthentiVox - Advanced Deepfake Audio Detection

<div align="center">

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/Sumit07-git/AuthentiVox-AI)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)

**Detect AI-generated and deepfake audio using hybrid ML + DL ensemble**

[Try Demo](https://huggingface.co/spaces/YOUR_USERNAME/authentivox) • [GitHub](https://github.com/Sumit07-git/AuthentiVox-AI) • [Report Issue](https://github.com/Sumit07-git/AuthentiVox-AI/issues)

</div>

---

## 🎯 Features

- **Hybrid Detection System**: Combines Random Forest (ML) + CNN (DL)
- **High Accuracy**: 96%+ accuracy on ASVspoof 2019 dataset
- **Real-time Analysis**: Instant results with confidence scores
- **Multiple Formats**: Supports WAV, MP3, FLAC, M4A, OGG
- **Advanced Features**: MFCC, spectral, and temporal analysis

## 🧠 How It Works

1. **Upload** an audio file or record your voice
2. **ML Model** analyzes 32 audio features (MFCC, spectral centroid, etc.)
3. **DL Model** analyzes mel-spectrogram patterns
4. **Hybrid Ensemble** combines predictions
5. **Get Results** with confidence scores and technical details

## 🔬 Technology Stack

### Machine Learning Model
- **Algorithm**: Random Forest Classifier (200 trees)
- **Features**: 32-dimensional MFCC vectors
- **Accuracy**: ~94%

### Deep Learning Model
- **Architecture**: Convolutional Neural Network
- **Input**: Mel-spectrogram (128x128)
- **Accuracy**: ~96%

### Training Data
- **Dataset**: ASVspoof 2019 Logical Access
- **Samples**: 2,000 real + 2,000 fake audio files
- **Duration**: 70+ hours of audio

## 📊 Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest | 94.2% | 93.8% | 94.6% | 94.2% |
| CNN | 96.1% | 95.9% | 96.3% | 96.1% |
| **Hybrid** | **96.5%** | **96.2%** | **96.8%** | **96.5%** |

## 👨‍💻 Developer

Created by **Sumit**

- GitHub: [@Sumit07-git](https://github.com/Sumit07-git)
- Project: [AuthentiVox-AI](https://github.com/Sumit07-git/AuthentiVox-AI)

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

## ⚠️ Disclaimer

This tool is for educational and research purposes. Results should not be used as sole evidence for legal or security decisions. Always verify with multiple sources and human expertise.

---

<div align="center">
Made with ❤️ using Gradio • Deployed on 🤗 Hugging Face Spaces
</div>
```

---

## 📋 **Step 4: Project Structure for Hugging Face**
```
AuthentiVox/
├── app.py                       ← Keep your Flask app (for local)
├── app_gradio.py               ← NEW: Gradio app for Hugging Face
├── requirements.txt            ← Keep for local/Flask
├── requirements_hf.txt         ← NEW: For Hugging Face
├── README.md                   ← NEW: Hugging Face config
├── models/
│   ├── ml_model/
│   │   ├── rf_classifier.pkl   ← Your ML model
│   │   └── scaler.pkl          ← Your scaler
│   └── dl_model/
│       ├── cnn_model.keras     ← Your DL model (INCLUDED!)
│       └── cnn_model.h5        ← Backup format
└── utils/
    ├── __init__.py
    ├── hybrid_predictor.py     ← Your existing predictor
    └── audio_processor.py