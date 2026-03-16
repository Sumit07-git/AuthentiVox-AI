import gradio as gr
from utils.hybrid_predictor import HybridPredictor
import os

# Load models
print("Loading models...")
predictor = HybridPredictor(
    ml_model_path='models/ml_model/rf_classifier.pkl',
    dl_model_path='models/dl_model/cnn_model.keras',
    scaler_path='models/ml_model/scaler.pkl'
)
print("Models loaded successfully!")

def analyze_audio(audio_file):
    """Analyze audio and return results"""
    
    if audio_file is None:
        return "Please upload an audio file"
    
    try:
        # Get prediction
        result = predictor.predict(audio_file)
        
        # Extract results
        is_fake = result['is_fake']
        confidence = result['confidence_score']
        duration = result.get('duration', 0)
        
        # Create simple output
        if is_fake:
            verdict = "🔴 FAKE AUDIO DETECTED"
            color = "#ff4444"
            message = "This audio appears to be AI-generated or manipulated."
        else:
            verdict = "✅ REAL AUDIO"
            color = "#44ff44"
            message = "This audio appears to be authentic."
        
        # Format output
        output = f"""
        <div style='padding: 30px; text-align: center;'>
            <h1 style='color: {color}; font-size: 2.5em;'>{verdict}</h1>
            <p style='font-size: 1.2em; margin: 20px 0;'>{message}</p>
            
            <div style='display: flex; justify-content: center; gap: 40px; margin: 30px 0;'>
                <div>
                    <p style='font-size: 0.9em; color: #666;'>Confidence</p>
                    <p style='font-size: 2em; font-weight: bold;'>{confidence:.1f}%</p>
                </div>
                <div>
                    <p style='font-size: 0.9em; color: #666;'>Duration</p>
                    <p style='font-size: 2em; font-weight: bold;'>{duration:.2f}s</p>
                </div>
            </div>
            
            <p style='color: #666; font-size: 0.9em; margin-top: 30px;'>
                Analysis powered by ML + DL hybrid model trained on ASVspoof 2019 dataset
            </p>
        </div>
        """
        
        return output
        
    except Exception as e:
        return f"""
        <div style='padding: 20px; background: #fee; border-radius: 10px;'>
            <h3>❌ Error</h3>
            <p>Failed to analyze audio: {str(e)}</p>
        </div>
        """

# Create interface
demo = gr.Interface(
    fn=analyze_audio,
    inputs=gr.Audio(
        type="filepath",
        label="Upload Audio File"
    ),
    outputs=gr.HTML(label="Results"),
    title="🎙️ AuthentiVox - Deepfake Audio Detection",
    description="Upload an audio file to detect if it's real or AI-generated. Supports WAV, MP3, FLAC, M4A formats.",
    theme="soft"
)

if __name__ == "__main__":
    demo.launch()