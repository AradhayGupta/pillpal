import cv2

try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None


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

    try:
        while True:
            if source == "picamera2":
                frame = cam.capture_array()
            else:
                ok, frame = cam.read()
                if not ok:
                    raise RuntimeError("Failed to read a frame from the camera.")

            frame = cv2.resize(frame, (width, height))
            cv2.circle(frame, middle, 10, (255, 0, 255), -1)
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