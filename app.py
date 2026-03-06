"""
Flask Application - Hybrid Deepfake Audio Detection System
OPTIMIZED FOR RENDER DEPLOYMENT
"""

import os
import sys
import logging
import traceback
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Setup logging to stdout for Render
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

app.config['SECRET_KEY'] = 'deepfake-detection-secret-key-2024'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3', 'flac', 'ogg'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global predictor - loaded ONCE on startup
predictor = None
predictor_error = None

def initialize_models():
    """Load models on startup - BLOCKS until complete"""
    global predictor, predictor_error
    
    logger.info("="*60)
    logger.info("INITIALIZING AI MODELS...")
    logger.info("="*60)
    sys.stdout.flush()
    
    try:
        # Check if model files exist
        ml_model = 'models/ml_model/rf_classifier.pkl'
        dl_model_keras = 'models/dl_model/cnn_model.keras'
        dl_model_h5 = 'models/dl_model/cnn_model.h5'
        scaler = 'models/ml_model/scaler.pkl'
        
        logger.info("Checking for model files...")
        logger.info(f"ML Model exists: {os.path.exists(ml_model)}")
        logger.info(f"DL Model (.keras) exists: {os.path.exists(dl_model_keras)}")
        logger.info(f"DL Model (.h5) exists: {os.path.exists(dl_model_h5)}")
        logger.info(f"Scaler exists: {os.path.exists(scaler)}")
        sys.stdout.flush()
        
        if not os.path.exists(ml_model):
            raise FileNotFoundError(f"ML model not found: {ml_model}")
        
        if not os.path.exists(scaler):
            raise FileNotFoundError(f"Scaler not found: {scaler}")
        
        # Import and initialize predictor
        from utils.hybrid_predictor import HybridPredictor
        
        logger.info("Loading HybridPredictor...")
        sys.stdout.flush()
        
        predictor = HybridPredictor()
        
        logger.info("✓ AI Models loaded successfully!")
        logger.info(f"  - ML Model: {'✓' if predictor.ml_model else '✗'}")
        logger.info(f"  - DL Model: {'✓' if predictor.dl_model else '✗'}")
        logger.info(f"  - Scaler: {'✓' if predictor.scaler else '✗'}")
        logger.info("="*60)
        sys.stdout.flush()
        
        return True
        
    except Exception as e:
        predictor_error = str(e)
        logger.error("="*60)
        logger.error("FAILED TO LOAD MODELS")
        logger.error(f"Error: {predictor_error}")
        logger.error("="*60)
        traceback.print_exc()
        sys.stdout.flush()
        return False

# LOAD MODELS ON STARTUP (not on first request)
logger.info("Starting server initialization...")
sys.stdout.flush()
models_loaded = initialize_models()

if not models_loaded:
    logger.warning("⚠ Server starting WITHOUT models loaded")
    logger.warning("⚠ /api/upload will return errors")
else:
    logger.info("✓ Server ready to accept requests")

sys.stdout.flush()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_audio_duration(filepath):
    try:
        import librosa
        y, sr = librosa.load(filepath, sr=None, duration=None)
        return librosa.get_duration(y=y, sr=sr)
    except Exception as e:
        logger.error(f"Duration error: {e}")
        return None

def clear_upload_folder():
    try:
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
    except Exception as e:
        logger.error(f"Clear folder error: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload_page():
    return render_template('upload.html')

@app.route('/history')
def history_page():
    return render_template('history.html')

@app.route('/api/health')
def health_check():
    try:
        status = {
            'status': 'healthy' if predictor else 'unhealthy',
            'models_loaded': {
                'predictor': predictor is not None,
                'ml_model': predictor.ml_model is not None if predictor else False,
                'dl_model': predictor.dl_model is not None if predictor else False,
                'scaler': predictor.scaler is not None if predictor else False
            },
            'error': predictor_error
        }
        return jsonify(status), 200
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/api/upload', methods=['POST', 'OPTIONS'])
def upload_file():
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        logger.info("Upload request received")
        
        # Check if models are loaded
        if predictor is None:
            logger.error("Models not loaded")
            return jsonify({
                'success': False,
                'error': 'AI models not available. Server may be starting up. Please try again in a moment.'
            }), 503
        
        # Validate file
        if 'audio_file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['audio_file']
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': f'Invalid format. Allowed: {", ".join(app.config["ALLOWED_EXTENSIONS"])}'
            }), 400
        
        logger.info(f"Processing: {file.filename}")
        
        # Save file
        clear_upload_folder()
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Check duration
        import librosa
        import numpy as np
        
        duration = get_audio_duration(filepath)
        if duration is None:
            return jsonify({'success': False, 'error': 'Invalid audio file'}), 400
        
        if duration > 60:
            return jsonify({
                'success': False,
                'error': f'Audio too long ({duration:.1f}s). Max 60 seconds.'
            }), 400
        
        logger.info(f"Duration: {duration:.2f}s")
        
        # Generate spectrogram
        spectrogram_path = None
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import librosa.display
            
            y, sr = librosa.load(filepath, sr=22050, duration=30)
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            spec_filename = f"spec_{filename.rsplit('.', 1)[0]}.png"
            spec_filepath = os.path.join(app.config['UPLOAD_FOLDER'], spec_filename)
            
            fig, ax = plt.subplots(figsize=(12, 4))
            librosa.display.specshow(mel_spec_db, sr=sr, hop_length=512, cmap='viridis', ax=ax)
            ax.axis('off')
            ax.set_frame_on(False)
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.margins(0, 0)
            plt.savefig(spec_filepath, dpi=100, bbox_inches='tight', pad_inches=0, facecolor='black')
            plt.close('all')
            
            spectrogram_path = f'/static/uploads/{spec_filename}'
            logger.info("Spectrogram generated")
        except Exception as e:
            logger.warning(f"Spectrogram failed: {e}")
        
        # Predict
        logger.info("Running prediction...")
        result = predictor.predict_hybrid(filepath, method='weighted_average')
        logger.info(f"Result: {result}")
        
        response = {
            'success': True,
            'filename': filename,
            'duration': round(duration, 2),
            'prediction': 'REAL' if result['hybrid_prediction'] == 1 else 'FAKE',
            'is_fake': result['hybrid_prediction'] == 0,
            'confidence_score': round(result['hybrid_confidence'] * 100, 2),
            'spectrogram_path': spectrogram_path
        }
        
        logger.info(f"SUCCESS: {response['prediction']} ({response['confidence_score']}%)")
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'success': False, 'error': 'File too large. Max 16MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"500 error: {e}")
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting server on port {port}...")
    app.run(debug=False, host='0.0.0.0', port=port)