"""
Flask Application - Hybrid Deepfake Audio Detection System
Professional multi-page web interface for audio deepfake detection
"""

import os
import sys
import shutil
import logging
import traceback
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
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


def get_predictor():
    """Lazy load the predictor - only loads when first request comes"""
    global predictor, predictor_error

    
    if predictor is not None:
        return predictor

    
    if predictor_error is not None:
        return None

    try:
        logger.info("=" * 50)
        logger.info("LOADING AI MODELS (first request)...")
        logger.info("=" * 50)
        sys.stdout.flush()

        
        model_dirs = ['models', 'models/ml_model', 'models/dl_model']
        for d in model_dirs:
            if os.path.exists(d):
                files = os.listdir(d)
                logger.info(f"Directory '{d}': {files}")
            else:
                logger.warning(f"Directory '{d}' NOT FOUND")
        sys.stdout.flush()

        from utils.hybrid_predictor import HybridPredictor
        predictor = HybridPredictor()
        logger.info("AI models loaded successfully!")
        sys.stdout.flush()
        return predictor

    except FileNotFoundError as e:
        predictor_error = f"Model files not found: {str(e)}"
        logger.error(f"ERROR: {predictor_error}")
        traceback.print_exc()
        sys.stdout.flush()
        return None

    except MemoryError as e:
        predictor_error = "Server ran out of memory loading models"
        logger.error(f"MEMORY ERROR: {predictor_error}")
        sys.stdout.flush()
        return None

    except Exception as e:
        predictor_error = f"Failed to load models: {str(e)}"
        logger.error(f"ERROR: {predictor_error}")
        traceback.print_exc()
        sys.stdout.flush()
        return None


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def get_audio_duration(filepath):
    """Get duration of audio file"""
    try:
        import librosa
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

    
    if request.method == 'OPTIONS':
        return '', 204

    try:
        logger.info("=" * 40)
        logger.info("UPLOAD REQUEST RECEIVED")
        logger.info("=" * 40)
        sys.stdout.flush()

        
        if 'audio_file' not in request.files:
            logger.error("No file in request")
            logger.info(f"Request files keys: {list(request.files.keys())}")
            sys.stdout.flush()
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400

        file = request.files['audio_file']

        
        if file.filename == '':
            logger.error("Empty filename")
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400

        
        if not allowed_file(file.filename):
            logger.error(f"Invalid file type: {file.filename}")
            return jsonify({
                'success': False,
                'error': f'Invalid file format. Allowed: {", ".join(app.config["ALLOWED_EXTENSIONS"])}'
            }), 400

        logger.info(f"Processing file: {file.filename}")
        sys.stdout.flush()

        
        clear_upload_folder()

        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        file_size = os.path.getsize(filepath)
        logger.info(f"File saved: {filepath} ({file_size} bytes)")
        sys.stdout.flush()

        
        logger.info("Loading audio file...")
        sys.stdout.flush()

        import librosa
        import numpy as np

        duration = get_audio_duration(filepath)
        if duration is None:
            logger.error("Invalid audio file")
            return jsonify({
                'success': False,
                'error': 'Invalid audio file or corrupted'
            }), 400

        
        if duration > 60:
            logger.error(f"Audio too long: {duration}s")
            return jsonify({
                'success': False,
                'error': f'Audio too long ({duration:.1f}s). Maximum is 60 seconds.'
            }), 400

        logger.info(f"Audio duration: {duration:.2f}s")
        sys.stdout.flush()

        
        spectrogram_path = None
        try:
            logger.info("Generating spectrogram...")
            sys.stdout.flush()

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
            plt.close('all')

            spectrogram_path = f'/static/uploads/{spec_filename}'
            logger.info(f"Spectrogram saved: {spectrogram_path}")
            sys.stdout.flush()

        except Exception as e:
            logger.warning(f"Could not generate spectrogram: {str(e)}")
            sys.stdout.flush()

        
        logger.info("Loading AI model...")
        sys.stdout.flush()

        current_predictor = get_predictor()

        if current_predictor is None:
            error_msg = predictor_error or 'AI models failed to load'
            logger.error(f"Predictor not available: {error_msg}")
            sys.stdout.flush()
            return jsonify({
                'success': False,
                'error': f'AI models not available: {error_msg}'
            }), 500

        
        logger.info("Running AI prediction...")
        sys.stdout.flush()

        result = current_predictor.predict_hybrid(filepath, method='weighted_average')

        logger.info(f"Prediction result: {result}")
        sys.stdout.flush()

        
        response_data = {
            'success': True,
            'filename': filename,
            'duration': round(duration, 2),
            'prediction': 'REAL' if result['hybrid_prediction'] == 1 else 'FAKE',
            'is_fake': result['hybrid_prediction'] == 0,
            'confidence_score': round(result['hybrid_confidence'] * 100, 2) if result['hybrid_confidence'] else 0,
            'spectrogram_path': spectrogram_path
        }

        logger.info(f"SUCCESS: {response_data['prediction']} ({response_data['confidence_score']}%)")
        sys.stdout.flush()

        return jsonify(response_data), 200

    except MemoryError:
        logger.error("OUT OF MEMORY during processing")
        sys.stdout.flush()
        return jsonify({
            'success': False,
            'error': 'Server ran out of memory. Try a smaller audio file.'
        }), 500

    except Exception as e:
        logger.error(f"UPLOAD ERROR: {str(e)}")
        traceback.print_exc()
        sys.stdout.flush()
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
            'predictor_error': predictor_error,
            'ml_model': predictor.ml_model is not None if predictor else False,
            'dl_model': predictor.dl_model is not None if predictor else False,
            'scaler': predictor.scaler is not None if predictor else False
        }

        all_ready = predictor is not None and all([
            models_status['ml_model'],
            models_status['dl_model'],
            models_status['scaler']
        ])

        return jsonify({
            'status': 'healthy' if all_ready else 'degraded',
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

        
        if os.path.exists('models'):
            for root, dirs, files in os.walk('models'):
                for f in files:
                    filepath = os.path.join(root, f)
                    try:
                        size = os.path.getsize(filepath)
                        files_info[filepath] = f"{size} bytes ({size / 1024 / 1024:.2f} MB)"
                    except Exception:
                        files_info[filepath] = "cannot read size"
        else:
            files_info['error'] = 'models/ directory does not exist'

        
        utils_files = []
        if os.path.exists('utils'):
            utils_files = os.listdir('utils')

        return jsonify({
            'models_dir_exists': os.path.exists('models'),
            'utils_dir_exists': os.path.exists('utils'),
            'utils_files': utils_files,
            'model_files': files_info,
            'total_model_files': len(files_info)
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
    print("\n" + "=" * 60)
    print("  HYBRID DEEPFAKE AUDIO DETECTION SYSTEM")
    print("=" * 60)
    print(f"\nModels Status: {'Loaded' if predictor else 'Will load on first request'}")
    print("\nStarting Flask server...")
    print("Access the application at: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60 + "\n")

    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)