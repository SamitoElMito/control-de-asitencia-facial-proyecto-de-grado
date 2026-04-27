import cv2
import pickle
from pathlib import Path
from paths import short_path

ROOT = Path(__file__).parent.parent
MODEL_PATH = ROOT / "models" / "modelo_face.xml"
ENCODER_PATH = ROOT / "models" / "label_encoder.pkl"

# LBPH distance: lower = closer match. Tune between 50–80.
CONFIDENCE_THRESHOLD = 80


def _load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo no encontrado en {MODEL_PATH}. Ejecuta train_model.py primero."
        )
    if not ENCODER_PATH.exists():
        raise FileNotFoundError(
            f"Encoder no encontrado en {ENCODER_PATH}. Ejecuta train_model.py primero."
        )
    rec = cv2.face.LBPHFaceRecognizer_create()
    rec.read(short_path(MODEL_PATH))
    with open(ENCODER_PATH, "rb") as f:
        enc = pickle.load(f)
    return rec, enc


try:
    _recognizer, _label_encoder = _load_model()
    _model_loaded = True
except (FileNotFoundError, cv2.error) as e:
    print(f"[recognizer] {e}")
    _recognizer, _label_encoder = None, None
    _model_loaded = False


def recognize_face(face_region):
    """
    Returns (name: str, confidence: float).
    confidence is the LBPH distance — lower means a better match.
    Returns ("Desconocido", confidence) when the face doesn't match anyone.
    Returns ("Sin modelo", 999.0) when the model hasn't been trained yet.
    """
    if not _model_loaded:
        return "Sin modelo", 999.0

    try:
        face_resized = cv2.equalizeHist(cv2.resize(face_region, (200, 200)))
    except Exception as e:
        print(f"[recognizer] Error al preprocesar: {e}")
        return "Desconocido", 999.0

    label, confidence = _recognizer.predict(face_resized)

    if confidence < CONFIDENCE_THRESHOLD:
        name = _label_encoder.inverse_transform([label])[0]
        return name, confidence

    return "Desconocido", confidence
