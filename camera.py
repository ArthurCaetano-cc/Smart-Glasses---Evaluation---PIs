"""
Abstração de câmera compartilhada entre os clientes da raspberry (main.py, identify.py):
usa Picamera2/libcamera para o módulo de câmera CSI (padrão em Bullseye/Bookworm) ou
OpenCV/V4L2 para webcams USB ou a stack de câmera legada.
"""
import os

import cv2
from dotenv import load_dotenv

try:
    from picamera2 import Picamera2
    HAS_PICAMERA2 = True
except ImportError:
    HAS_PICAMERA2 = False

load_dotenv()

CAMERA_INDEX = int(os.getenv('CAMERA_INDEX', '0'))
CAMERA_BACKEND = os.getenv('CAMERA_BACKEND', 'auto')  # 'auto', 'picamera2' ou 'opencv'
CAMERA_WIDTH = int(os.getenv('CAMERA_WIDTH', '640'))
CAMERA_HEIGHT = int(os.getenv('CAMERA_HEIGHT', '480'))


class OpenCVCamera:
    """Câmera via OpenCV/V4L2 -- webcams USB ou stack de câmera legada do Raspberry Pi."""

    def __init__(self, index):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Não foi possível abrir a câmera (index={index})")

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()


class Picamera2Camera:
    """Câmera via Picamera2/libcamera -- módulo de câmera CSI do Raspberry Pi (Bullseye/Bookworm)."""

    def __init__(self, width, height):
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(main={"format": "BGR888", "size": (width, height)})
        self.picam2.configure(config)
        self.picam2.start()

    def read(self):
        return True, self.picam2.capture_array()

    def release(self):
        self.picam2.stop()
        self.picam2.close()


def create_camera():
    backend = CAMERA_BACKEND
    if backend == 'auto':
        backend = 'picamera2' if HAS_PICAMERA2 else 'opencv'

    if backend == 'picamera2':
        if not HAS_PICAMERA2:
            raise RuntimeError(
                "CAMERA_BACKEND=picamera2, mas o pacote picamera2 não está instalado. "
                "Instale com 'sudo apt install -y python3-picamera2' (recriando o venv com "
                "--system-site-packages) ou defina CAMERA_BACKEND=opencv."
            )
        print("Usando backend de câmera: picamera2 (libcamera)")
        return Picamera2Camera(CAMERA_WIDTH, CAMERA_HEIGHT)

    print(f"Usando backend de câmera: OpenCV (index={CAMERA_INDEX})")
    return OpenCVCamera(CAMERA_INDEX)
