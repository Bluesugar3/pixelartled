"""Simple camera preview.
Press 'q' to quit.

Dependencies:
- Windows/macOS/Linux (USB webcam): pip install opencv-python
- Raspberry Pi (Arducam/RPi Cam v2): sudo apt install python3-picamera2; pip install opencv-python
"""
import cv2
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    Picamera2 = Any  # type: ignore

# Prefer Pi Camera (Picamera2) on Raspberry Pi; fallback to USB/DirectShow/Video4Linux
try:
    from picamera2 import Picamera2  # type: ignore
    from picamera2.previews import Preview  # type: ignore
    _HAVE_PICAMERA2 = True
except Exception:
    Picamera2 = None  # type: ignore
    _HAVE_PICAMERA2 = False


def main() -> None:
    picam = None  # type: ignore[assignment]
    cap = None  # type: ignore[assignment]
    headless = os.environ.get("DISPLAY") is None and os.environ.get("WAYLAND_DISPLAY") is None

    if _HAVE_PICAMERA2:
        try:
            # If no CSI cameras are detected, fall back to OpenCV.
            if not Picamera2.global_camera_info():
                raise RuntimeError("No Pi cameras detected")
            picam = Picamera2()
            RES = (640, 480)
            cfg = picam.create_video_configuration(main={"size": RES})
            picam.configure(cfg)
            if headless:
                try:
                    picam.start_preview(Preview.DRM)
                    print("Picamera2 DRM preview started (headless). Press Ctrl+C to quit.")
                except Exception as e:
                    print(f"Failed to start DRM preview: {e}")
            picam.start()
        except Exception as e:
            print(f"Picamera2 unavailable ({e}); falling back to OpenCV (/dev/video0).")
            picam = None
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("ERROR: Could not open camera. No Pi camera and no /dev/video0.")
                return
    else:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("ERROR: Could not open camera.")
            return

    try:
        notified = False
        while True:
            if picam is not None:
                try:
                    frame = picam.capture_array()  # RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    ok = True
                except Exception:
                    ok = False
            else:
                ok, frame = cap.read()

            if not ok:
                print("WARNING: Failed to read frame.")
                break

            # Only show a window if a GUI is available
            if not (picam is not None and headless):
                cv2.imshow("Camera", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                # Headless with DRM preview active; no imshow. Print a one-time hint.
                if not notified:
                    print("Headless mode: live preview on HDMI. Press Ctrl+C to exit.")
                    notified = True
    finally:
        if picam is not None:
            picam.stop()
        elif cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
