import os
import time
from datetime import datetime

import cv2
from picamera2 import Picamera2


def open_camera():
    if Picamera2 is not None:
        try:
            cam = Picamera2()
            cam.configure(cam.create_video_configuration(main={"format": "XRGB8888", "size": (640, 480)}))
            cam.start()
            return cam, "picamera2"
        except Exception as exc:
            print(f"Picamera2 initialization failed: {exc}")

    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        return cap, "webcam"

    raise RuntimeError(
        "No supported camera device is available. On Raspberry Pi, ensure picamera2 is installed and the camera is enabled."
    )


def main():
    cam, source = open_camera()
    width, height = 640, 480
    middle = (width // 2, height // 2)
    save_dir = os.path.join(os.path.dirname(__file__), "captures")
    os.makedirs(save_dir, exist_ok=True)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if face_cascade.empty():
        raise RuntimeError("Face cascade could not be loaded. Check that OpenCV's haarcascades data is installed.")

    face_seen = False
    last_save_time = 0.0
    save_interval_seconds = 2.0

    try:
        while True:
            if source == "picamera2":
                frame = cam.capture_array()
            else:
                ok, frame = cam.read()
                if not ok:
                    raise RuntimeError("Failed to read a frame from the camera.")

            frame = cv2.resize(frame, (width, height))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
            )

            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Face detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                now = time.time()
                if now - last_save_time >= save_interval_seconds:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    photo_path = os.path.join(save_dir, f"face_{timestamp}.jpg")
                    success = cv2.imwrite(photo_path, frame)
                    if success:
                        print(f"Saved face snapshot to {photo_path}")
                        last_save_time = now
                    else:
                        print("Failed to save face snapshot")

                face_seen = True
            else:
                face_seen = False

            cv2.circle(frame, middle, 10, (255, 0, 255), -1)
            cv2.putText(frame, "Press q to quit", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow("Camera", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        print("Camera stream stopped.")
    finally:
        if source == "picamera2":
            cam.stop()
        else:
            cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()