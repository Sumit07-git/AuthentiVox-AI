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

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/Sumit07-git/AuthentiVox-AI)
[![HuggingFace](https://img.shields.io/badge/🤗-Space-yellow)](https://huggingface.co/spaces/sumit0788/authentivox)

Detects AI-generated and deepfake audio with **96%+ accuracy** using hybrid ML/DL approach.

## 🎯 Features

- 🤖 **Hybrid Architecture**: Random Forest (94%) + CNN (96%)
- ⚡ **Fast Predictions**: <3 seconds per audio
- 📊 **Visual Analysis**: Real-time spectrogram generation
- 🎨 **User-Friendly**: Clean, responsive web interface
- 🔒 **Privacy-Focused**: No data retention

## 🛠️ Tech Stack

**Backend:**
- Python, Flask, TensorFlow, scikit-learn

**Audio Processing:**
- Librosa, NumPy, SciPy

**Frontend:**
- HTML5, CSS3, JavaScript

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest (ML) | 94.1% | 93.8% | 94.4% | 94.1% |
| CNN (DL) | 96.3% | 95.9% | 96.7% | 96.3% |
| **Hybrid Ensemble** | **96.4%** | **95.8%** | **96.2%** | **96.0%** |

## 🎓 Dataset

Trained on **ASVspoof 2019** dataset:
- 4,000+ audio samples
- Balanced real/fake distribution
- Multiple audio formats

## 🚀 How to Use

1. **Upload** an audio file (WAV, MP3, FLAC)
2. **Wait** for analysis (~3 seconds)
3. **View** prediction with confidence score and spectrogram

## 💻 Local Development

```bash
# Clone repository
git clone https://github.com/Sumit07-git/AuthentiVox-AI.git
cd AuthentiVox-AI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run application
python app.py
```

## 📁 Project Structure
AuthentiVox-AI/
├── app.py                 # Flask application
├── requirements.txt       # Python dependencies
├── Dockerfile            # Container configuration
├── utils/                # Utility modules
│   ├── feature_extractor.py
│   ├── spectrogram_generator.py
│   └── hybrid_predictor.py
├── models/               # Trained models
│   ├── ml_model/
│   └── dl_model/
├── templates/            # HTML templates
└── static/              # CSS, JS, images

## 🔬 Technical Details

**Feature Extraction:**
- 13 MFCCs (Mel-frequency cepstral coefficients)
- Spectral features (centroid, rolloff, bandwidth)
- Zero-crossing rate
- Chroma features

**Model Architecture:**
- **ML**: Random Forest (200 estimators)
- **DL**: 4-layer CNN with batch normalization
- **Ensemble**: Weighted averaging (40% ML, 60% DL)

## 📈 Future Improvements

- [ ] Real-time audio streaming analysis
- [ ] Multi-language support
- [ ] API rate limiting
- [ ] Mobile app version
- [ ] Batch processing

## 👨‍💻 Author

**Sumit Kumar**
- GitHub: [@Sumit07-git](https://github.com/Sumit07-git)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details

## 🙏 Acknowledgments

- ASVspoof 2019 dataset providers
- TensorFlow and scikit-learn communities
- Hugging Face for hosting

---

**⭐ Star this repo if you find it helpful!**