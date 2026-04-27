---
title: AuthentiVox AI
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# 🎙️ AuthentiVox AI - Deepfake Audio Detection

[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-yellow)](https://huggingface.co/spaces/sumit0788/authentivox)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/Sumit07-git/AuthentiVox-AI)

Detect AI-generated and deepfake audio with **96%+ accuracy** using a hybrid ML/DL approach.

## 🎯 Features

- 🤖 **Hybrid AI Architecture**: Combines Random Forest (ML) and CNN (Deep Learning)
- ⚡ **Fast Detection**: Results in under 3 seconds
- 📊 **Visual Analysis**: Real-time mel-spectrogram generation
- 🎨 **User-Friendly Interface**: Clean, responsive web design
- 🔒 **Privacy-Focused**: No data retention or logging

## 🛠️ Technology Stack

**Backend:**
- Python 3.11
- Flask (Web Framework)
- TensorFlow 2.15 (Deep Learning)
- scikit-learn 1.5 (Machine Learning)

**Audio Processing:**
- Librosa (Feature Extraction)
- NumPy (Numerical Computing)
- Matplotlib (Visualization)

**Frontend:**
- HTML5, CSS3, JavaScript
- Responsive Design

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest (ML) | 94.1% | 93.8% | 94.4% | 94.1% |
| CNN (Deep Learning) | 96.3% | 95.9% | 96.7% | 96.3% |
| **Hybrid Ensemble** | **96.4%** | **95.8%** | **96.2%** | **96.0%** |

## 🎓 Training Dataset

- **Source**: ASVspoof 2019 Challenge Dataset
- **Samples**: 4,000+ audio files
- **Distribution**: Balanced real/fake audio
- **Formats**: WAV, FLAC (converted to consistent format)

## 🚀 How to Use

1. **Upload** an audio file (WAV, MP3, FLAC supported)
2. **Click** "Analyze Audio"
3. **Wait** ~3 seconds for processing
4. **View Results**:
   - Prediction: REAL or FAKE
   - Confidence Score
   - Visual Spectrogram
   - ML vs DL comparison

## 🔬 Technical Details

### Feature Extraction (ML Model)
- 13 MFCCs (Mel-frequency cepstral coefficients)
- Spectral Centroid, Rolloff, Bandwidth
- Zero-Crossing Rate
- Chroma Features
- Temporal Statistics

### Deep Learning Architecture (CNN)
- **Input**: 128x128 Mel-Spectrogram
- **Architecture**: 4 Convolutional Layers
- **Regularization**: Batch Normalization, Dropout (0.5)
- **Output**: Binary Classification (Real/Fake)

### Ensemble Method
- **Strategy**: Weighted Averaging
- **Weights**: 40% ML + 60% DL
- **Rationale**: DL better at frequency patterns, ML better at temporal features

## 💻 Local Development

```bashClone repository
git clone https://github.com/Sumit07-git/AuthentiVox-AI.git
cd AuthentiVox-AICreate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activateInstall dependencies
pip install -r requirements.txtRun application
python app.py

Visit `http://localhost:5000` in your browser.

## 📁 Project StructureAuthentiVox-AI/
├── app.py                      # Flask application
├── Dockerfile                  # Container configuration
├── requirements.txt            # Python dependencies
├── utils/                      # Utility modules
│   ├── feature_extractor.py   # Audio feature extraction
│   ├── spectrogram_generator.py # Mel-spectrogram generation
│   └── hybrid_predictor.py    # Ensemble prediction
├── models/                     # Trained models
│   ├── ml_model/
│   │   ├── rf_classifier.pkl  # Random Forest model
│   │   └── scaler.pkl         # Feature scaler
│   └── dl_model/
│       ├── cnn_model.h5       # CNN model (primary)
│       └── cnn_model.keras    # CNN model (backup)
├── templates/                  # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── upload.html
│   └── history.html
└── static/                     # Static assets
├── css/
├── js/
└── images/

## 🎯 Use Cases

### Security & Fraud Prevention
- Verify caller identity in phone banking
- Detect voice phishing (vishing) attempts
- Validate audio evidence in investigations

### Media Verification
- Authenticate news audio clips
- Verify interview recordings
- Detect manipulated podcast content

### Content Moderation
- Flag AI-generated audio on platforms
- Verify user-submitted audio content
- Maintain content authenticity standards

## 📈 Future Improvements

- [ ] Real-time streaming audio analysis
- [ ] Multi-language support
- [ ] REST API with authentication
- [ ] Batch processing for multiple files
- [ ] Mobile application (iOS/Android)
- [ ] Model retraining pipeline
- [ ] Enhanced visualization dashboard

## 👨‍💻 Author

**Sumit Kumar**

- GitHub: [@Sumit07-git](https://github.com/Sumit07-git)
- LinkedIn: [Connect with me](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **ASVspoof Challenge** organizers for the dataset
- **TensorFlow** and **scikit-learn** communities
- **Hugging Face** for hosting infrastructure
- All contributors and testers

## 📚 References

- ASVspoof 2019: [Official Challenge](http://www.asvspoof.org/)
- Research Paper: [Link to your paper if applicable]
- Blog Post: [Technical deep-dive if you write one]

## 🐛 Issues & Contributions

Found a bug or want to contribute?

1. **Report Issues**: [GitHub Issues](https://github.com/Sumit07-git/AuthentiVox-AI/issues)
2. **Pull Requests**: Contributions welcome!
3. **Discussions**: [GitHub Discussions](https://github.com/Sumit07-git/AuthentiVox-AI/discussions)

---

**⭐ If you find this project helpful, please star the repository!**

**🔗 Try it live:** [AuthentiVox on Hugging Face](https://huggingface.co/spaces/sumit0788/authentivox)