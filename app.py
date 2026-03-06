"""
Flask Application - Hybrid Deepfake Audio Detection System
Professional multi-page web interface for audio deepfake detection
"""

import os
import shutil
import logging
import traceback
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename
import librosa
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS

app.config['SECRET_KEY'] = 'deepfake-detection-secret-key-2024'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3', 'flac', 'ogg'}

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize predictor
predictor = None

def initialize_predictor():
    """Initialize the hybrid predictor with error handling"""
    global predictor
    try:
        logger.info("🔄 Initializing hybrid predictor...")
        from utils.hybrid_predictor import HybridPredictor
        predictor = HybridPredictor()
        logger.info("✅ Hybrid predictor initialized successfully")
        return True
    except FileNotFoundError as e:
        logger.error(f"❌ Model files not found: {str(e)}")
        logger.error("Please ensure model files are in the correct directories:")
        logger.error("  - models/ml_model/")
        logger.error("  - models/dl_model/")
        return False
    except Exception as e:
        logger.error(f"❌ Error initializing predictor: {str(e)}")
        logger.error(traceback.format_exc())
        return False

# Try to initialize predictor on startup
logger.info("🚀 Starting AuthentiVox Application...")
models_loaded = initialize_predictor()

if not models_loaded:
    logger.warning("⚠️  Application started WITHOUT models loaded")
    logger.warning("⚠️  Training or model download required")

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_audio_duration(filepath):
    """Get duration of audio file"""
    try:
        y, sr = librosa.load(filepath, sr=None, duration=None)
        duration = librosa.get_duration(y=y, sr=sr)
        return duration
    except Exception as e:
        logger.error(f"Error getting audio duration: {str(e)}")
        return None

def clear_upload_folder():
    """Clear all files in upload folder"""
    try:
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
    except Exception as e:
        logger.error(f"Error clearing upload folder: {str(e)}")

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
    """History page"""
    return render_template('history.html')

@app.route('/api/upload', methods=['POST', 'OPTIONS'])
def upload_file():
    """Handle file upload and analysis"""
    
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        logger.info("📁 Upload request received")
        
        # Check if models are loaded
        if predictor is None:
            logger.error("❌ Predictor not initialized")
            return jsonify({
                'success': False,
                'error': 'AI models are not loaded. Please train the models first or contact support.'
            }), 500
        
        # Check if file exists
        if 'audio_file' not in request.files:
            logger.error("❌ No file in request")
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400
        
        file = request.files['audio_file']
        
        # Check if filename is empty
        if file.filename == '':
            logger.error("❌ Empty filename")
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Validate file type
        if not allowed_file(file.filename):
            logger.error(f"❌ Invalid file type: {file.filename}")
            return jsonify({
                'success': False,
                'error': f'Invalid file format. Allowed formats: {", ".join(app.config["ALLOWED_EXTENSIONS"])}'
            }), 400
        
        logger.info(f"📄 Processing file: {file.filename}")
        
        # Clear old uploads
        clear_upload_folder()
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logger.info(f"💾 File saved to: {filepath}")
        
        # Get audio duration
        duration = get_audio_duration(filepath)
        if duration is None:
            logger.error("❌ Invalid audio file")
            return jsonify({
                'success': False,
                'error': 'Invalid audio file or corrupted'
            }), 400
        
        # Check duration limit
        if duration > 60:
            logger.error(f"❌ Audio too long: {duration}s")
            return jsonify({
                'success': False,
                'error': f'Audio too long ({duration:.1f}s). Maximum duration is 60 seconds.'
            }), 400
        
        logger.info(f"⏱️  Audio duration: {duration:.2f}s")
        
        # Generate spectrogram
        spectrogram_path = None
        try:
            logger.info("🎨 Generating spectrogram...")
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import librosa.display
            
            # Load audio
            y, sr = librosa.load(filepath, sr=22050, duration=30)
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Save spectrogram
            spec_filename = f"spec_{filename.rsplit('.', 1)[0]}.png"
            spec_filepath = os.path.join(app.config['UPLOAD_FOLDER'], spec_filename)
            
            fig, ax = plt.subplots(figsize=(12, 4))
            librosa.display.specshow(
                mel_spec_db,
                sr=sr,
                hop_length=512,
                cmap='viridis',
                ax=ax
            )
            
            ax.axis('off')
            ax.set_frame_on(False)
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.margins(0, 0)
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())
            
            plt.savefig(spec_filepath, 
                       dpi=100, 
                       bbox_inches='tight', 
                       pad_inches=0,
                       facecolor='black')
            plt.close()
            
            spectrogram_path = f'/static/uploads/{spec_filename}'
            logger.info(f"✅ Spectrogram generated: {spectrogram_path}")
            
        except Exception as e:
            logger.warning(f"⚠️  Could not generate spectrogram: {str(e)}")
        
        # Predict using hybrid model
        logger.info("🤖 Running AI prediction...")
        result = predictor.predict_hybrid(filepath, method='weighted_average')
        logger.info(f"✅ Prediction complete: {result}")
        
        # Prepare response
        response_data = {
            'success': True,
            'filename': filename,
            'duration': round(duration, 2),
            'prediction': 'REAL' if result['hybrid_prediction'] == 1 else 'FAKE',
            'is_fake': result['hybrid_prediction'] == 0,
            'confidence_score': round(result['hybrid_confidence'] * 100, 2) if result['hybrid_confidence'] else 0,
            'spectrogram_path': spectrogram_path
        }
        
        logger.info(f"📊 Response: {response_data['prediction']} ({response_data['confidence_score']}%)")
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"❌ Error in upload_file: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f'Processing error: {str(e)}'
        }), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    try:
        models_status = {
            'predictor_initialized': predictor is not None,
            'ml_model': predictor.ml_model is not None if predictor else False,
            'dl_model': predictor.dl_model is not None if predictor else False,
            'scaler': predictor.scaler is not None if predictor else False
        }
        
        all_ready = all(models_status.values())
        
        return jsonify({
            'status': 'healthy',
            'models_loaded': models_status,
            'all_models_ready': all_ready
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/debug/files')
def debug_files():
    """Debug endpoint to check which files exist"""
    try:
        files_info = {}
        
        # Check models directory
        if os.path.exists('models'):
            for root, dirs, files in os.walk('models'):
                for f in files:
                    filepath = os.path.join(root, f)
                    size = os.path.getsize(filepath)
                    files_info[filepath] = f"{size} bytes"
        
        return jsonify({
            'files_found': len(files_info) > 0,
            'files': files_info
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

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
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🎵 HYBRID DEEPFAKE AUDIO DETECTION SYSTEM")
    print("="*60)
    print(f"\nModels Status: {'✅ Loaded' if predictor else '❌ Not Loaded'}")
    print("\nStarting Flask server...")
    print("Access the application at: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    app.run(debug=False, host='0.0.0.0', port=5000)