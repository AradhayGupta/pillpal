import os
import time
from datetime import datetime

import cv2

try:
    from picamera2 import Picamera2
except Exception:
    Picamera2 = None


def open_camera():
    if Picamera2 is not None:
        try:
            cam = Picamera2()
            cam.configure(cam.create_video_configuration(main={"format": "XRGB8888", "size": (640, 480)}))
            cam.start()
            return cam
        except Exception as exc:
            raise RuntimeError(f"Picamera2 initialization failed: {exc}") from exc

    raise RuntimeError(
        "Picamera2 is not available. On Raspberry Pi, ensure picamera2 is installed and the camera is enabled."
    )


def load_face_cascade():
    candidates = []

    if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
        candidates.append(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))

    cv2_module_path = getattr(cv2, "__file__", "")
    if cv2_module_path:
        module_dir = os.path.dirname(cv2_module_path)
        candidates.extend(
            [
                os.path.join(module_dir, "data", "haarcascade_frontalface_default.xml"),
                os.path.join(module_dir, "haarcascade_frontalface_default.xml"),
            ]
        )

    candidates.extend(
        [
            "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
            "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
            "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
        ]
    )

    if os.environ.get("VIRTUAL_ENV"):
        candidates.append(os.path.join(os.environ["VIRTUAL_ENV"], "share", "opencv4", "haarcascades", "haarcascade_frontalface_default.xml"))

    seen = set()
    for path in candidates:
        if path and path not in seen and os.path.exists(path):
            seen.add(path)
            cascade = cv2.CascadeClassifier(path)
            if not cascade.empty():
                return cascade

    raise RuntimeError(
        "Face cascade could not be loaded. Install OpenCV with haarcascades data or check the package installation."
    )


def load_profile_cascade():
    """Same search strategy as load_face_cascade, but for the side-profile model."""
    candidates = []

    if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
        candidates.append(os.path.join(cv2.data.haarcascades, "haarcascade_profileface.xml"))

    cv2_module_path = getattr(cv2, "__file__", "")
    if cv2_module_path:
        module_dir = os.path.dirname(cv2_module_path)
        candidates.extend(
            [
                os.path.join(module_dir, "data", "haarcascade_profileface.xml"),
                os.path.join(module_dir, "haarcascade_profileface.xml"),
            ]
        )

    candidates.extend(
        [
            "/usr/share/opencv4/haarcascades/haarcascade_profileface.xml",
            "/usr/local/share/opencv4/haarcascades/haarcascade_profileface.xml",
            "/usr/share/opencv/haarcascades/haarcascade_profileface.xml",
        ]
    )

    for path in candidates:
        if path and os.path.exists(path):
            cascade = cv2.CascadeClassifier(path)
            if not cascade.empty():
                return cascade

    return None  # profile detection is a bonus, not required


def load_eye_cascade():
    """Same search strategy as load_face_cascade, but for the eye model."""
    candidates = []

    if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
        candidates.append(os.path.join(cv2.data.haarcascades, "haarcascade_eye.xml"))

    cv2_module_path = getattr(cv2, "__file__", "")
    if cv2_module_path:
        module_dir = os.path.dirname(cv2_module_path)
        candidates.extend(
            [
                os.path.join(module_dir, "data", "haarcascade_eye.xml"),
                os.path.join(module_dir, "haarcascade_eye.xml"),
            ]
        )

    candidates.extend(
        [
            "/usr/share/opencv4/haarcascades/haarcascade_eye.xml",
            "/usr/local/share/opencv4/haarcascades/haarcascade_eye.xml",
            "/usr/share/opencv/haarcascades/haarcascade_eye.xml",
        ]
    )

    for path in candidates:
        if path and os.path.exists(path):
            cascade = cv2.CascadeClassifier(path)
            if not cascade.empty():
                return cascade

    return None  # eye confirmation is a bonus, not required


def is_face_like(w, h):
    """Reject boxes that are wildly off from a face aspect ratio (very loose)."""
    aspect = w / float(h)
    return 0.55 <= aspect <= 1.8


def main():
    cam = open_camera()
    width, height = 640, 480
    middle = (width // 2, height // 2)
    save_dir = os.path.join(os.path.dirname(__file__), "captures")
    os.makedirs(save_dir, exist_ok=True)

    face_cascade = load_face_cascade()
    profile_cascade = load_profile_cascade()  # may be None if not found on this system
    eye_cascade = load_eye_cascade()          # may be None if not found on this system

    last_save_time = 0.0
    save_interval_seconds = 2.0

    # --- Speed settings ---
    detect_every_n_frames = 2   # run detection on every Nth frame
    detect_scale = 0.5          # run detection on a downscaled copy (0.5 = half size)
    frame_count = 0
    last_box = None             # reused between detection frames

    # --- Accuracy settings ---
    # Two-tier strategy instead of one global minNeighbors value:
    #   - STRICT pass: high minNeighbors, trusted immediately (very unlikely
    #     to be a curtain, but also unlikely to catch angled/turned faces)
    #   - LOOSE pass: low minNeighbors, catches angled/marginal faces, but a
    #     candidate from this pass is only accepted if an eye is also found
    #     inside the box. Curtains/blinds essentially never have eye-like
    #     features, so this lets us loosen recall without losing precision.
    strict_min_neighbors = 9
    loose_min_neighbors = 4
    min_size_px = 50            # in downscaled-frame pixels
    consecutive_hits_required = 1
    hit_streak = 0

    try:
        while True:
            frame = cam.capture_array()
            frame = cv2.resize(frame, (width, height))

            frame_count += 1
            run_detection = (frame_count % detect_every_n_frames == 0)

            if run_detection:
                small = cv2.resize(frame, None, fx=detect_scale, fy=detect_scale)
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

                # STRICT pass: high confidence, trusted outright
                strict_faces = list(
                    face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.1,
                        minNeighbors=strict_min_neighbors,
                        minSize=(min_size_px, min_size_px),
                    )
                )

                # LOOSE pass: catches angled/marginal faces, but needs an
                # eye found inside the box before we trust it.
                loose_candidates = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=loose_min_neighbors,
                    minSize=(min_size_px, min_size_px),
                )

                confirmed_loose = []
                for (fx, fy, fw, fh) in loose_candidates:
                    # skip if this candidate is basically already in strict_faces
                    if any(abs(fx - sx) < 10 and abs(fy - sy) < 10 for sx, sy, sw, sh in strict_faces):
                        continue
                    if eye_cascade is not None:
                        roi = gray[fy:fy + fh, fx:fx + fw]
                        eyes = eye_cascade.detectMultiScale(
                            roi, scaleFactor=1.1, minNeighbors=5, minSize=(12, 12)
                        )
                        if len(eyes) >= 1:
                            confirmed_loose.append((fx, fy, fw, fh))
                    # if no eye cascade available, loose candidates are dropped
                    # (falls back to strict-only behavior)

                faces = strict_faces + confirmed_loose

                # Profile faces (turned heads) skip the eye check since a
                # side profile often only shows one eye or none clearly;
                # the profile cascade's own minNeighbors is the filter here.
                if profile_cascade is not None:
                    profile_min_neighbors = 7  # moderate: profile cascade is naturally more selective
                    faces.extend(
                        profile_cascade.detectMultiScale(
                            gray,
                            scaleFactor=1.1,
                            minNeighbors=profile_min_neighbors,
                            minSize=(min_size_px, min_size_px),
                        )
                    )
                    flipped = cv2.flip(gray, 1)
                    flipped_faces = profile_cascade.detectMultiScale(
                        flipped,
                        scaleFactor=1.1,
                        minNeighbors=profile_min_neighbors,
                        minSize=(min_size_px, min_size_px),
                    )
                    img_w = gray.shape[1]
                    for (fx, fy, fw, fh) in flipped_faces:
                        faces.append((img_w - fx - fw, fy, fw, fh))

                # Filter out non-face-shaped detections (curtains, blinds, etc.)
                faces = [f for f in faces if is_face_like(f[2], f[3])]

                if len(faces) > 0:
                    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
                    # scale coordinates back up to full frame size
                    scale_back = 1.0 / detect_scale
                    last_box = (
                        int(x * scale_back),
                        int(y * scale_back),
                        int(w * scale_back),
                        int(h * scale_back),
                    )
                    hit_streak += 1
                else:
                    hit_streak = 0
                    last_box = None

            face_confirmed = last_box is not None and hit_streak >= consecutive_hits_required

            if face_confirmed:
                x, y, w, h = last_box
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
            else:
                cv2.putText(frame, "No face detected", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.circle(frame, middle, 10, (255, 0, 255), -1)
            cv2.putText(frame, "Press q to quit", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow("Camera", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        print("Camera stream stopped.")
    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()