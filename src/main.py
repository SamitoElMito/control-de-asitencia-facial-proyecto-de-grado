import cv2
import time
import collections
from pathlib import Path
from recognizer import recognize_face
from paths import short_path

ROOT = Path(__file__).parent.parent
CASCADE_PATH = short_path(ROOT / "models" / "cascades" / "haarcascade_frontalface_alt2.xml")

# Must match train_model.py so the crop geometry is identical to training time
CASCADE_SCALE = 1.1
CASCADE_NEIGHBORS = 5
CASCADE_MIN_SIZE = (60, 60)

COLOR_KNOWN = (0, 200, 0)    # green
COLOR_UNKNOWN = (0, 0, 220)  # red
COLOR_TEXT = (255, 255, 255) # white

# Number of consecutive frames used to vote on the final label (reduces flickering)
SMOOTHING_FRAMES = 7


def _draw_label(frame, text, x, y, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.7, 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = max(0, min(x, frame.shape[1] - tw - 8))
    top = max(0, y - th - baseline - 6)
    cv2.rectangle(frame, (x, top), (x + tw + 6, y), color, cv2.FILLED)
    cv2.putText(frame, text, (x + 3, y - baseline - 2), font, scale, COLOR_TEXT, thickness)


def _majority(history):
    """Return the label that appears most often in the history deque."""
    counts = collections.Counter(history)
    return counts.most_common(1)[0][0]


def main():
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if face_cascade.empty():
        print(f"Error: no se pudo cargar el clasificador desde:\n  {CASCADE_PATH}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo acceder a la cámara.")
        return

    print("Camara activada. Presiona 'q' para salir.")

    # history[slot] holds the last SMOOTHING_FRAMES predictions for that face slot
    # We use a simple single-slot history (one dominant face per frame)
    history = collections.deque(maxlen=SMOOTHING_FRAMES)
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=CASCADE_SCALE,
            minNeighbors=CASCADE_NEIGHBORS,
            minSize=CASCADE_MIN_SIZE,
        )

        if len(faces) == 0:
            # No face detected this frame — keep history stable (don't push Unknown)
            smoothed_name = _majority(history) if history else "Desconocido"
            display_conf = None
        else:
            # Use the largest detected face for smoothing
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            face_region = gray[y : y + h, x : x + w]
            name, confidence = recognize_face(face_region)

            history.append(name)
            smoothed_name = _majority(history)
            display_conf = confidence if smoothed_name == name else None

            for (fx, fy, fw, fh) in faces:
                known = smoothed_name not in ("Desconocido", "Sin modelo")
                color = COLOR_KNOWN if known else COLOR_UNKNOWN
                cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), color, 2)

            known = smoothed_name not in ("Desconocido", "Sin modelo")
            color = COLOR_KNOWN if known else COLOR_UNKNOWN
            label = f"{smoothed_name}  {int(display_conf)}" if (known and display_conf is not None) else smoothed_name
            _draw_label(frame, label, x, y, color)

        now = time.time()
        fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
        prev_time = now
        cv2.putText(
            frame, f"FPS {fps:.1f}", (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1,
        )

        cv2.imshow("Reconocimiento Facial", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
