"""
Flask Application - Hybrid Deepfake Audio Detection System
Professional multi-page web interface for audio deepfake detection
"""

import os
import shutil
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import librosa
import numpy as np
from utils.hybrid_predictor import HybridPredictor

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'deepfake-detection-secret-key-2024'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3', 'flac', 'ogg'}

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize hybrid predictor
try:
    predictor = HybridPredictor()
    print("✓ Hybrid predictor initialized successfully")
except Exception as e:
    print(f"⚠ Warning: Could not initialize predictor - {str(e)}")
    predictor = None


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def get_audio_duration(filepath):
    """Get audio duration in seconds"""
    try:
        y, sr = librosa.load(filepath, sr=None, duration=None)
        duration = librosa.get_duration(y=y, sr=sr)
        return duration
    except:
        return None


def clear_upload_folder():
    """Clear old files from upload folder"""
    try:
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
    except Exception as e:
        print(f"Error clearing upload folder: {str(e)}")


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/upload')
def upload_page():
    """Upload page"""
    return render_template('upload.html')


@app.route('/history')
def history_page():
    """History and reports page"""
    return render_template('history.html')


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload and prediction"""
    
    # Check if predictor is initialized
    if predictor is None:
        return jsonify({
            'success': False,
            'error': 'Models not loaded. Please train the models first.'
        }), 500
    
    # Check if file is present
    if 'audio_file' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No file provided'
        }), 400
    
    file = request.files['audio_file']
    
    # Check if file is selected
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'No file selected'
        }), 400
    
    # Check file extension
    if not allowed_file(file.filename):
        return jsonify({
            'success': False,
            'error': f'Invalid file format. Allowed formats: {", ".join(app.config["ALLOWED_EXTENSIONS"])}'
        }), 400
    
    try:
        # Clear old files
        clear_upload_folder()
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Check audio duration
        duration = get_audio_duration(filepath)
        if duration is None:
            return jsonify({
                'success': False,
                'error': 'Invalid audio file or corrupted'
            }), 400
        
        # Limit duration to 60 seconds
        if duration > 60:
            return jsonify({
                'success': False,
                'error': f'Audio too long ({duration:.1f}s). Maximum duration is 60 seconds.'
            }), 400
        
        # Generate spectrogram visualization
        spectrogram_path = None
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            import librosa.display
            
            # Generate mel spectrogram
            y, sr = librosa.load(filepath, sr=22050, duration=30)
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Save spectrogram image - CLEAN VERSION (no axes)
            spec_filename = f"spec_{filename.rsplit('.', 1)[0]}.png"
            spec_filepath = os.path.join(app.config['UPLOAD_FOLDER'], spec_filename)
            
            # Create figure with no axes or labels
            fig, ax = plt.subplots(figsize=(12, 4))
            
            # Display spectrogram
            librosa.display.specshow(
                mel_spec_db,
                sr=sr,
                hop_length=512,
                cmap='viridis',
                ax=ax
            )
            
            # Remove all axes, labels, and whitespace
            ax.axis('off')
            ax.set_frame_on(False)
            
            # Remove margins and padding
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.margins(0, 0)
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())
            
            # Save with no padding - clean spectrogram only
            plt.savefig(spec_filepath, 
                       dpi=100, 
                       bbox_inches='tight', 
                       pad_inches=0,
                       facecolor='black')
            plt.close()
            
            spectrogram_path = f'/static/uploads/{spec_filename}'
        except Exception as e:
            print(f"Warning: Could not generate spectrogram - {str(e)}")
        
        # Get prediction
        result = predictor.predict_hybrid(filepath, method='weighted_average')
        
        # Prepare simplified response - only hybrid results
        response_data = {
            'success': True,
            'filename': filename,
            'duration': round(duration, 2),
            'prediction': 'REAL' if result['hybrid_prediction'] == 1 else 'FAKE',
            'is_fake': result['hybrid_prediction'] == 0,
            'confidence_score': round(result['hybrid_confidence'] * 100, 2) if result['hybrid_confidence'] else 0,
            'spectrogram_path': spectrogram_path
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Processing error: {str(e)}'
        }), 500


@app.route('/api/health')
def health_check():
    """API health check"""
    models_status = {
        'ml_model': predictor.ml_model is not None if predictor else False,
        'dl_model': predictor.dl_model is not None if predictor else False,
        'scaler': predictor.scaler is not None if predictor else False
    }
    
    return jsonify({
        'status': 'healthy',
        'models_loaded': models_status,
        'all_models_ready': all(models_status.values())
    })


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum size is 16MB.'
    }), 413


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return render_template('index.html'), 404


@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🎵 HYBRID DEEPFAKE AUDIO DETECTION SYSTEM")
    print("="*60)
    print("\nStarting Flask server...")
    print("Access the application at: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    # Debug mode disabled to prevent auto-reload issues
    app.run(debug=False, host='0.0.0.0', port=5000)