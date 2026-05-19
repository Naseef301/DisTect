
import numpy as np
import librosa
import joblib
import tensorflow as tf
import os
import warnings
warnings.filterwarnings('ignore')


class DistressPredictor:

    def __init__(self,
                 model_path='models/distress_detection_model_v2.keras',
                 scaler_path='models/feature_scaler_v2.joblib',
                 label_encoder_path='models/label_encoder_v2.joblib'):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.label_encoder_path = label_encoder_path
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self._load_model_and_scaler()

    def _load_model_and_scaler(self):
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            if not os.path.exists(self.scaler_path):
                raise FileNotFoundError(f"Scaler file not found: {self.scaler_path}")
            if not os.path.exists(self.label_encoder_path):
                raise FileNotFoundError(f"Label encoder file not found: {self.label_encoder_path}")

            print(f"Loading model from {self.model_path}...")
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
            print("Model loaded successfully!")

            print(f"Loading scaler from {self.scaler_path}...")
            self.scaler = joblib.load(self.scaler_path)
            print("Scaler loaded successfully!")

            print(f"Loading label encoder from {self.label_encoder_path}...")
            self.label_encoder = joblib.load(self.label_encoder_path)
            print("Label encoder loaded successfully!")
            print(f"Emotion classes: {self.label_encoder.classes_}")

        except Exception as e:
            print(f"Error loading model or preprocessing objects: {e}")
            raise

    def extract_features(self, file_path):
        """
        275-dimensional feature vector matching model_v2 training:
          MFCC mean (40) + Delta (40) + Delta2 (40)
          + Chroma (12) + Mel (128) + Contrast (7)
          + Tonnetz (6) + ZCR (1) + RMS (1) = 275
        """
        try:
            X, sr = librosa.load(file_path, sr=22050, res_type='soxr_hq')

            result = []

            # MFCC + Delta + Delta2
            mfcc   = librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40)
            delta  = librosa.feature.delta(mfcc)
            delta2 = librosa.feature.delta(mfcc, order=2)
            result.append(np.mean(mfcc.T,   axis=0))   # 40
            result.append(np.mean(delta.T,  axis=0))   # 40
            result.append(np.mean(delta2.T, axis=0))   # 40

            # Chroma
            stft   = np.abs(librosa.stft(X))
            chroma = librosa.feature.chroma_stft(S=stft, sr=sr, n_chroma=12)
            result.append(np.mean(chroma.T, axis=0))   # 12

            # Mel spectrogram
            mel = librosa.feature.melspectrogram(y=X, sr=sr, n_mels=128, fmax=8000)
            result.append(np.mean(mel.T, axis=0))      # 128

            # Spectral Contrast
            contrast = librosa.feature.spectral_contrast(S=stft, sr=sr, n_bands=6)
            result.append(np.mean(contrast.T, axis=0)) # 7

            # Tonnetz
            harmonic = librosa.effects.harmonic(X)
            tonnetz  = librosa.feature.tonnetz(y=harmonic, sr=sr)
            result.append(np.mean(tonnetz.T, axis=0))  # 6

            # ZCR + RMS
            result.append(np.array([np.mean(librosa.feature.zero_crossing_rate(X))]))  # 1
            result.append(np.array([np.mean(librosa.feature.rms(y=X))]))               # 1

            return np.concatenate(result)  # → 275

        except Exception as e:
            print(f"Error extracting features from {file_path}: {e}")
            raise

    def predict(self, file_path):
        try:
            print(f"Extracting features from {file_path}...")
            features = self.extract_features(file_path)

            if features is None or features.size == 0:
                raise ValueError(f"Feature extraction failed for {file_path}")

            scaled   = self.scaler.transform(features.reshape(1, -1))
            reshaped = np.expand_dims(scaled, axis=-1)  # (1, 275, 1) for Conv1D

            print("Making prediction...")
            distress_prob, emotion_probs = self.model.predict(reshaped, verbose=0)

            distress_probability = float(distress_prob[0][0])
            distress_label = "Distress" if distress_probability > 0.5 else "Non-Distress"

            emotion_probabilities = {
                emotion: float(emotion_probs[0][i])
                for i, emotion in enumerate(self.label_encoder.classes_)
            }
            predicted_emotion = self.label_encoder.classes_[np.argmax(emotion_probs[0])]

            return {
                'distress_probability': distress_probability,
                'distress_label':       distress_label,
                'emotion_probabilities':emotion_probabilities,
                'predicted_emotion':    predicted_emotion,
                'filename':             os.path.basename(file_path)
            }

        except Exception as e:
            print(f"Error during prediction: {e}")
            raise

    def predict_batch(self, file_paths):
        results = []
        for file_path in file_paths:
            try:
                results.append(self.predict(file_path))
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                results.append({'filename': os.path.basename(file_path), 'error': str(e)})
        return results


def main():
    print("=== Voice Distress Detection - Prediction Engine ===\n")
    try:
        predictor = DistressPredictor()
    except Exception as e:
        print(f"Failed to initialize predictor: {e}")
        return

    audio_file = input("Audio file path (or Enter to skip): ").strip()
    if audio_file and os.path.exists(audio_file):
        try:
            result = predictor.predict(audio_file)
            print(f"\nFile: {result['filename']}")
            print(f"Distress: {result['distress_label']} ({result['distress_probability']*100:.2f}%)")
            print(f"Emotion:  {result['predicted_emotion']}")
            for emo, prob in sorted(result['emotion_probabilities'].items(), key=lambda x: x[1], reverse=True):
                print(f"  {emo.capitalize()}: {prob*100:.2f}%")
        except Exception as e:
            print(f"Error: {e}")
    elif audio_file:
        print(f"File not found: {audio_file}")


if __name__ == "__main__":
    main()

