import gradio as gr
import subprocess
import os

# Start Flask in background
subprocess.Popen(["python", "app.py"])

# Gradio wrapper
def analyze_audio(audio_file):
    import requests
    
    # Wait for Flask to start
    import time
    time.sleep(2)
    
    # Call Flask API
    with open(audio_file, 'rb') as f:
        files = {'audio_file': f}
        response = requests.post('http://localhost:5000/api/upload', files=files)
    
    if response.status_code == 200:
        data = response.json()
        return f"{data['prediction']} ({data['confidence_score']}%)"
    else:
        return "Error analyzing audio"

# Create Gradio interface
demo = gr.Interface(
    fn=analyze_audio,
    inputs=gr.Audio(type="filepath", label="Upload Audio"),
    outputs=gr.Textbox(label="Result"),
    title="AuthentiVox - Deepfake Audio Detector",
    description="Upload an audio file to detect if it's real or AI-generated"
)

demo.launch(server_name="0.0.0.0", server_port=7860)