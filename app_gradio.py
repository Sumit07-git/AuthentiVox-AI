import gradio as gr
from utils.hybrid_predictor import HybridPredictor
import os

# Load models
print("=" * 60)
print("Loading models...")
print("=" * 60)

try:
    predictor = HybridPredictor(
        ml_model_path='models/ml_model/rf_classifier.pkl',
        dl_model_path='models/dl_model/cnn_model.keras',
        scaler_path='models/ml_model/scaler.pkl'
    )
    
    ml_status = "✓" if predictor.ml_model else "✗"
    dl_status = "✓" if predictor.dl_model else "✗"
    
    print(f"ML Model: {ml_status}")
    print(f"DL Model: {dl_status}")
    print("=" * 60)
    
except Exception as e:
    print(f"Error loading models: {e}")
    predictor = None

def analyze_audio(audio_file):
    """Analyze audio and return results"""
    
    if audio_file is None:
        return "⚠️ Please upload an audio file"
    
    if predictor is None:
        return "❌ Models not loaded. Please check logs."
    
    try:
        # Get prediction
        result = predictor.predict_hybrid(audio_file, method='weighted_average')
        
        # Format results
        is_fake = result['hybrid_prediction'] == 0
        confidence = result['hybrid_confidence'] * 100
        
        if is_fake:
            verdict = "🔴 FAKE AUDIO DETECTED"
            color = "#ff4444"
            message = "This audio appears to be AI-generated or manipulated."
        else:
            verdict = "✅ REAL AUDIO"
            color = "#44ff44"
            message = "This audio appears to be authentic."
        
        # Model info
        model_used = "Hybrid (ML + DL)" if predictor.dl_model else "ML Only"
        
        output = f"""
        <div style='padding: 30px; text-align: center; font-family: Arial;'>
            <h1 style='color: {color}; font-size: 2.5em; margin: 20px 0;'>{verdict}</h1>
            <p style='font-size: 1.2em; color: #333; margin: 20px 0;'>{message}</p>
            
            <div style='display: flex; justify-content: center; gap: 40px; margin: 30px 0; flex-wrap: wrap;'>
                <div style='background: #f0f0f0; padding: 20px; border-radius: 10px; min-width: 150px;'>
                    <p style='font-size: 0.9em; color: #666; margin: 0;'>Confidence</p>
                    <p style='font-size: 2em; font-weight: bold; margin: 10px 0;'>{confidence:.1f}%</p>
                </div>
                <div style='background: #f0f0f0; padding: 20px; border-radius: 10px; min-width: 150px;'>
                    <p style='font-size: 0.9em; color: #666; margin: 0;'>Model Used</p>
                    <p style='font-size: 1.2em; font-weight: bold; margin: 10px 0;'>{model_used}</p>
                </div>
            </div>
            
            <div style='margin-top: 30px; padding: 20px; background: #f9f9f9; border-radius: 8px; border-left: 4px solid #667eea;'>
                <p style='margin: 0; color: #666; font-size: 0.95em; text-align: left;'>
                    <strong>Analysis powered by:</strong><br>
                    • Random Forest (ML) - 94% accuracy<br>
                    • CNN (DL) - 96% accuracy<br>
                    • Trained on ASVspoof 2019 dataset (4,000+ samples)
                </p>
            </div>
        </div>
        """
        
        return output
        
    except Exception as e:
        return f"""
        <div style='padding: 20px; background: #fee; border-radius: 10px; border-left: 4px solid #f00;'>
            <h3 style='color: #d00; margin-top: 0;'>❌ Analysis Error</h3>
            <p style='color: #333;'>{str(e)}</p>
        </div>
        """

# Create Gradio interface
demo = gr.Interface(
    fn=analyze_audio,
    inputs=gr.Audio(
        type="filepath",
        label="🎤 Upload Audio File",
        sources=["upload", "microphone"]
    ),
    outputs=gr.HTML(label="📊 Analysis Results"),
    title="🎙️ AuthentiVox - Deepfake Audio Detection",
    description="""
    ### AI-Powered Deepfake Audio Detection
    
    Upload an audio file or record your voice to check authenticity using hybrid ML + DL models.
    
    **Supported Formats:** WAV, MP3, FLAC, M4A, OGG  
    **Technology:** Random Forest (ML) + CNN (DL) hybrid ensemble  
    **Accuracy:** 96%+ on ASVspoof 2019 dataset
    """,
    article="""
    ---
    
    ### About AuthentiVox
    
    AuthentiVox combines traditional machine learning and deep learning to detect AI-generated audio:
    
    - **ML Model:** Random Forest with MFCC feature extraction (94% accuracy)
    - **DL Model:** Convolutional Neural Network on mel-spectrograms (96% accuracy)
    - **Hybrid Ensemble:** Weighted combination for optimal performance (96%+ accuracy)
    
    **Dataset:** ASVspoof 2019 Logical Access (4,000+ samples)  
    **Developer:** Sumit  
    **GitHub:** [AuthentiVox-AI](https://github.com/Sumit07-git/AuthentiVox-AI)
    
    ---
    
    ⚠️ **Disclaimer:** This tool is for educational purposes. Results should not be used as sole evidence for critical decisions.
    """,
    theme="soft",
    examples=None,
    cache_examples=False
)

if __name__ == "__main__":
    demo.launch()