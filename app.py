



import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import numpy as np

from predict import DistressPredictor

# app = Flask(__name__)

app = Flask(__name__,
            static_folder='UI',
            static_url_path='/UI',
            template_folder='UI/templates')
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac', 'm4a', 'webm'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
predictor = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

with app.app_context():
    try:
        print("Loading model and preprocessing objects...")
        predictor = DistressPredictor(
            model_path='models/distress_detection_model_v2.keras',
            scaler_path='models/feature_scaler_v2.joblib',
            label_encoder_path='models/label_encoder_v2.joblib'
        )
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    if predictor is None:
        return jsonify({'status': 'unhealthy', 'message': 'Model not loaded'}), 503
    return jsonify({'status': 'healthy', 'message': 'Service is running',
                    'model_loaded': predictor.model is not None}), 200

@app.route('/predict', methods=['POST'])
def predict():
    global predictor
    if predictor is None:
        return jsonify({'error': 'Model not loaded.'}), 503
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided in request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': f'File type not allowed. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

    temp_path = None
    try:
        filename  = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)
        print(f"Processing file: {filename}")
        result = predictor.predict(temp_path)
        if os.path.exists(temp_path):
            os.remove(temp_path)
        response = {
            'success': True,
            'filename': result['filename'],
            'distress': {
                'label':       result['distress_label'],
                'probability': round(result['distress_probability'], 4),
                'percentage':  round(result['distress_probability'] * 100, 2)
            },
            'emotions': {
                emotion: {'probability': round(prob, 4), 'percentage': round(prob * 100, 2)}
                for emotion, prob in result['emotion_probabilities'].items()
            },
            'predicted_emotion': result['predicted_emotion']
        }
        print(f"Prediction successful: {result['distress_label']}")
        return jsonify(response), 200

    except Exception as e:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    required_files = {
        'Model':         'models/distress_detection_model_v2.keras',
        'Scaler':        'models/feature_scaler_v2.joblib',
        'Label Encoder': 'models/label_encoder_v2.joblib'
    }
    for name, path in required_files.items():
        if not os.path.exists(path):
            print(f"ERROR: {name} not found at '{path}'")
            exit(1)

    print("Starting Voice Distress Detection Server...")
    app.run(host='0.0.0.0', port=5000, debug=False)

