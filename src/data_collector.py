import cv2
import os
from pathlib import Path
from paths import short_path

ROOT = Path(__file__).parent.parent
CASCADE_PATH = short_path(ROOT / "models" / "cascades" / "haarcascade_frontalface_alt2.xml")


def extract_faces_from_video(
    video_path: str,
    person_name: str,
    save_every_n: int = 3,
    max_images: int = 1000,
):
    """
    Reads a video file, detects the largest face in every Nth frame,
    crops it, and saves it to data/<person_name>/.
    Only frames where a face is detected are saved.
    """
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if face_cascade.empty():
        print(f"Error: no se pudo cargar el clasificador desde {CASCADE_PATH}")
        return

    output_folder = ROOT / "data" / person_name
    os.makedirs(output_folder, exist_ok=True)

    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: no se pudo abrir el video en {video_path}")
        return

    frame_count = 0
    saved = 0

    while saved < max_images:
        ret, frame = video.read()
        if not ret:
            break

        if frame_count % save_every_n == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)  # matches train_model.py / main.py
            )

            if len(detected) > 0:
                # Use the largest face found in the frame
                x, y, w, h = max(detected, key=lambda f: f[2] * f[3])
                face_crop = gray[y : y + h, x : x + w]
                face_resized = cv2.resize(face_crop, (200, 200))

                img_name = str(output_folder / f"{frame_count}.jpg")
                cv2.imwrite(img_name, face_resized)
                saved += 1
                print(f"  [{saved}/{max_images}] Guardado: {img_name}")

        frame_count += 1

    video.release()
    print(f"\nFinalizado. {saved} imagenes guardadas en {output_folder}")


if __name__ == "__main__":
    # Edit PERSON_NAME and VIDEO_FILE before running
    PERSON_NAME = "Samuel Jimenez"
    VIDEO_FILE = str(ROOT / "data" / "videos" / "Samuel_Jimenez.mp4")

    extract_faces_from_video(VIDEO_FILE, PERSON_NAME)
