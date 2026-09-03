"""
Cliente de exibição (raspberry): conecta ao WebSocket de reconhecimento facial da API
(ws/recognize) e exibe, com OpenCV, os frames da câmera com as caixas delimitadoras e o
resultado do reconhecimento (nome + confiança) desenhados sobre cada rosto detectado.

Protocolo (ver api/routes/websocket.py): a cada frame JPEG binário enviado, o servidor
responde com um único JSON {"faces": [...]} antes do próximo frame poder ser enviado --
por isso o laço abaixo é estritamente send/recv em série, sem overlap.
"""
import json
import os
import time

import cv2
import websocket
from dotenv import load_dotenv

try:
    from picamera2 import Picamera2
    HAS_PICAMERA2 = True
except ImportError:
    HAS_PICAMERA2 = False

load_dotenv()

SERVER_WS_URL = os.getenv('SERVER_WS_URL', 'ws://localhost:5000/ws/recognize')
CAMERA_INDEX = int(os.getenv('CAMERA_INDEX', '0'))
CAMERA_BACKEND = os.getenv('CAMERA_BACKEND', 'auto')  # 'auto', 'picamera2' ou 'opencv'
CAMERA_WIDTH = int(os.getenv('CAMERA_WIDTH', '640'))
CAMERA_HEIGHT = int(os.getenv('CAMERA_HEIGHT', '480'))
JPEG_QUALITY = int(os.getenv('JPEG_QUALITY', '80'))
RECONNECT_DELAY_SECONDS = 2

AUTHORIZED_COLOR = (0, 200, 0)
UNAUTHORIZED_COLOR = (0, 0, 220)


def draw_faces(frame, faces):
    for face in faces:
        x1, y1, x2, y2 = face['bbox']
        authorized = face.get('authorized', False)
        matches = face.get('matches') or []
        top = matches[0] if matches else None
        color = AUTHORIZED_COLOR if authorized else UNAUTHORIZED_COLOR

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{top['name']} ({top['confidence']:.1f}%)" if top else "Desconhecido"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_y = max(y1, th + baseline + 4)
        cv2.rectangle(frame, (x1, label_y - th - baseline - 4), (x1 + tw + 4, label_y), color, -1)
        cv2.putText(frame, label, (x1 + 2, label_y - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


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


def run():
    cap = create_camera()

    try:
        while True:
            ws = None
            try:
                print(f"Conectando em {SERVER_WS_URL}...")
                ws = websocket.create_connection(SERVER_WS_URL, timeout=5)
                print("Conectado.")

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("Falha ao capturar frame da câmera.")
                        break

                    ok, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                    if not ok:
                        continue

                    ws.send_binary(buffer.tobytes())
                    response = json.loads(ws.recv())
                    draw_faces(frame, response.get('faces', []))

                    cv2.imshow('Smart Glasses - Evaluation', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        return
            except (websocket.WebSocketException, ConnectionError, OSError) as e:
                print(f"Conexão com o servidor perdida ({e}). Reconectando em {RECONNECT_DELAY_SECONDS}s...")
                time.sleep(RECONNECT_DELAY_SECONDS)
            finally:
                if ws is not None:
                    ws.close()
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        run()
    except KeyboardInterrupt:
        pass
