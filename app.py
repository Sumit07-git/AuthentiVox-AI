"""
Flask Application - FIXED VERSION
Changes:
  1. Fixed confidence calculation (p_real → prediction-confidence)
  2. Updated debug field structure for frontend
"""

import os
import sys
import logging
import traceback
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

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

predictor = None
predictor_error = None


def initialize_models():
    """Load models on startup."""
    global predictor, predictor_error
    
    logger.info("=" * 60)
    logger.info("INITIALIZING AI MODELS...")
    logger.info("=" * 60)
    sys.stdout.flush()
    
    try:
        ml_model_primary = 'models/ml_model/rf_pipeline.pkl'
        ml_model_alias = 'models/ml_model/rf_classifier.pkl'
        dl_model_keras = 'models/dl_model/cnn_model.keras'
        dl_model_h5 = 'models/dl_model/cnn_model.h5'
        scaler = 'models/ml_model/scaler.pkl'
        
        ml_model_exists = os.path.exists(ml_model_primary) or os.path.exists(ml_model_alias)
        ml_model_path = ml_model_primary if os.path.exists(ml_model_primary) else ml_model_alias
        
        logger.info("Checking for model files...")
        logger.info(f"ML Model (pipeline): {os.path.exists(ml_model_primary)}")
        logger.info(f"ML Model (alias):    {os.path.exists(ml_model_alias)}")
        logger.info(f"DL Model (.keras):   {os.path.exists(dl_model_keras)}")
        logger.info(f"DL Model (.h5):      {os.path.exists(dl_model_h5)}")
        logger.info(f"Scaler:              {os.path.exists(scaler)}")
        sys.stdout.flush()
        
        if not ml_model_exists:
            raise FileNotFoundError(f"ML model not found at {ml_model_primary} or {ml_model_alias}")
        
        from utils.hybrid_predictor import HybridPredictor
        
        logger.info("Loading HybridPredictor...")
        sys.stdout.flush()
        
        predictor = HybridPredictor(ml_model_path=ml_model_path)
        
        logger.info("✓ AI Models loaded successfully!")
        logger.info(f"  - ML Model: {'✓' if predictor.ml_model else '✗'}")
        logger.info(f"  - DL Model: {'✓' if predictor.dl_model else '✗'}")
        logger.info(f"  - Scaler:   {'✓' if predictor.scaler else '✗'}")
        logger.info("=" * 60)
        sys.stdout.flush()
        
        return True
        
    except Exception as e:
        predictor_error = str(e)
        logger.error("=" * 60)
        logger.error("FAILED TO LOAD MODELS")
        logger.error(f"Error: {predictor_error}")
        logger.error("=" * 60)
        traceback.print_exc()
        sys.stdout.flush()
        return False


logger.info("Starting server initialization...")
sys.stdout.flush()
models_loaded = initialize_models()

if not models_loaded:
    logger.warning("⚠ Server starting WITHOUT models loaded")
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


def _prediction_confidence_percent(p_real, prediction):
    """
    Convert p_real → confidence in the prediction (0-100%).
    
    Args:
        p_real: Probability of being REAL [0, 1]
        prediction: 0 (FAKE) or 1 (REAL)
    
    Returns:
        confidence_percent: Confidence in the prediction (0-100%)
    """
    if p_real is None:
        return None
    
    if prediction == 1:
        # Predicted REAL → confidence = p_real
        return round(p_real * 100, 2)
    else:
        # Predicted FAKE → confidence = 1 - p_real
        return round((1.0 - p_real) * 100, 2)


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
        
        if predictor is None:
            return jsonify({
                'success': False,
                'error': 'AI models not available. Please try again.'
            }), 503
        
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
        
        clear_upload_folder()
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
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
        logger.info(f"Raw result: {result}")
        
        if result['hybrid_prediction'] is None or result['hybrid_confidence'] is None:
            logger.error("Both models failed to produce a prediction")
            return jsonify({
                'success': False,
                'error': 'Audio analysis failed. File may be corrupted.'
            }), 500
        
        # ✅ FIX: Use hybrid_confidence (which is p_real) to calculate prediction confidence
        hybrid_pred = result['hybrid_prediction']
        hybrid_p_real = result['hybrid_confidence']
        
        is_fake = (hybrid_pred == 0)
        
        # Confidence in the prediction
        if hybrid_pred == 1:
            # Predicted REAL → confidence = p_real
            confidence_score = round(hybrid_p_real * 100, 2)
        else:
            # Predicted FAKE → confidence = 1 - p_real
            confidence_score = round((1.0 - hybrid_p_real) * 100, 2)
        
        # ✅ FIX: Convert ML/DL p_real to prediction-confidence for frontend
        ml_pred = result['ml_prediction']
        dl_pred = result['dl_prediction']
        ml_p_real = result['ml_confidence']
        dl_p_real = result['dl_confidence']
        
        ml_conf_pct = _prediction_confidence_percent(ml_p_real, ml_pred)
        dl_conf_pct = _prediction_confidence_percent(dl_p_real, dl_pred)
        
        response = {
            'success': True,
            'filename': filename,
            'duration': round(duration, 2),
            'prediction': 'FAKE' if is_fake else 'REAL',
            'is_fake': is_fake,
            'confidence_score': confidence_score,
            'spectrogram_path': spectrogram_path,
            'debug': {
                'ml_prediction': ml_pred,
                'ml_confidence': ml_conf_pct,  # Confidence in ML prediction (%)
                'dl_prediction': dl_pred,
                'dl_confidence': dl_conf_pct,  # Confidence in DL prediction (%)
                'method': result['method']
            }
        }
        
        logger.info(
            f"SUCCESS: {response['prediction']} ({response['confidence_score']}%) "
            f"[ML={ml_pred}@{ml_conf_pct}%, DL={dl_pred}@{dl_conf_pct}%, method={result['method']}]"
        )
        
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
    port = int(os.environ.get('PORT', 7860))
    logger.info(f"Starting server on port {port}...")
    app.run(debug=False, host='0.0.0.0', port=port)