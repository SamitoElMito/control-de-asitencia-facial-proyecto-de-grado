import cv2
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
from pathlib import Path
from paths import short_path

ROOT = Path(__file__).parent.parent
CASCADE_PATH = short_path(ROOT / "models" / "cascades" / "haarcascade_frontalface_alt2.xml")
DATA_PATH = ROOT / "data"
MODEL_OUTPUT = short_path(ROOT / "models" / "modelo_face.xml")
ENCODER_OUTPUT = ROOT / "models" / "label_encoder.pkl"

# Must match main.py so the crop geometry is identical at inference time
CASCADE_SCALE = 1.1
CASCADE_NEIGHBORS = 5
CASCADE_MIN_SIZE = (60, 60)


def _largest_face(detected_faces):
    return max(detected_faces, key=lambda f: f[2] * f[3])


def _augment(face):
    """Return the original face plus three augmented variants."""
    bright = cv2.add(face, 20)
    dark = cv2.subtract(face, 20)
    flipped = cv2.flip(face, 1)
    return [face, bright, dark, flipped]


def train():
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if face_cascade.empty():
        print(f"Error: no se pudo cargar el clasificador desde {CASCADE_PATH}")
        return

    faces, labels = [], []

    people = [p for p in os.listdir(DATA_PATH) if (DATA_PATH / p).is_dir()]
    if not people:
        print(f"No se encontraron carpetas de personas en {DATA_PATH}")
        return

    for person_name in sorted(people):
        person_folder = DATA_PATH / person_name
        images = list(person_folder.iterdir())
        found = 0

        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Already a face crop — use directly
            if gray.shape[0] <= 250 and gray.shape[1] <= 250:
                face = cv2.equalizeHist(cv2.resize(gray, (200, 200)))
                for variant in _augment(face):
                    faces.append(variant)
                    labels.append(person_name)
                found += 1
                continue

            detected = face_cascade.detectMultiScale(
                gray,
                scaleFactor=CASCADE_SCALE,
                minNeighbors=CASCADE_NEIGHBORS,
                minSize=CASCADE_MIN_SIZE,
            )
            if len(detected) == 0:
                continue

            x, y, w, h = _largest_face(detected)
            face = cv2.equalizeHist(cv2.resize(gray[y : y + h, x : x + w], (200, 200)))
            for variant in _augment(face):
                faces.append(variant)
                labels.append(person_name)
            found += 1

        print(f"  {person_name}: {found} imagenes base → {found * 4} muestras (con augmentation)")

    if not faces:
        print("No se encontraron caras para entrenar. Revisa la carpeta data/.")
        return

    print(f"\nTotal de muestras: {len(faces)} de {len(people)} persona(s)")

    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    # radius=2, neighbors=16 captures a wider local neighbourhood than the default (1, 8)
    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=16, grid_x=8, grid_y=8)
    recognizer.train(faces, np.array(labels_encoded))

    recognizer.save(MODEL_OUTPUT)
    with open(ENCODER_OUTPUT, "wb") as f:
        pickle.dump(label_encoder, f)

    print(f"Modelo guardado en:  {MODEL_OUTPUT}")
    print(f"Encoder guardado en: {ENCODER_OUTPUT}")
    print("Listo. Ejecuta main.py para probar el reconocimiento.")


if __name__ == "__main__":
    train()
