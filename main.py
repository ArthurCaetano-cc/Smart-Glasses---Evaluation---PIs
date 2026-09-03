"""
Cliente de exibição (raspberry): conecta ao WebSocket de reconhecimento facial da API
(ws/recognize) e exibe, com OpenCV, os frames da câmera com as caixas delimitadoras e o
resultado do reconhecimento (nome + confiança) desenhados sobre cada rosto detectado.

Protocolo (ver api/routes/websocket.py): a cada frame JPEG binário enviado, o servidor
responde com um único JSON {"faces": [...]} antes do próximo frame poder ser enviado --
por isso o envio para o servidor é estritamente send/recv em série, sem overlap. Como o
processamento no servidor é bem mais lento que a captura de câmera, esse laço de
rede roda numa thread separada (recognition_worker) alimentada sempre com o frame mais
recente (latest_frame), enquanto a thread principal exibe a câmera ao vivo sem esperar
pela rede -- as caixas desenhadas usam o último resultado recebido (latest_faces), que
pode estar um pouco atrasado em relação ao frame exibido, mas sem travar o preview.
"""
import json
import os
import threading
import time

import cv2
import websocket
from dotenv import load_dotenv

from camera import create_camera

load_dotenv()

SERVER_WS_URL = os.getenv('SERVER_WS_URL', 'ws://localhost:5000/ws/recognize')
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


class LatestValue:
    """Guarda só o valor mais recente escrito, descartando os anteriores -- usado para
    passar frames/resultados entre a thread de captura/exibição e a de rede sem que uma
    espere a outra."""

    def __init__(self, initial=None):
        self._lock = threading.Lock()
        self._value = initial

    def set(self, value):
        with self._lock:
            self._value = value

    def get(self):
        with self._lock:
            return self._value


def recognition_worker(latest_frame, latest_faces, stop_event):
    """Roda em thread separada: mantém a conexão WebSocket com o servidor e, em loop,
    envia sempre o frame mais recente disponível (latest_frame) e atualiza o resultado
    mais recente (latest_faces) assim que a resposta chega. Nunca acumula frames
    atrasados -- se o servidor demorar, o próximo envio já pega o frame mais novo."""
    while not stop_event.is_set():
        ws = None
        try:
            print(f"Conectando em {SERVER_WS_URL}...")
            ws = websocket.create_connection(SERVER_WS_URL, timeout=5)
            print("Conectado.")

            while not stop_event.is_set():
                frame = latest_frame.get()
                if frame is None:
                    time.sleep(0.01)
                    continue

                ok, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if not ok:
                    continue

                ws.send_binary(buffer.tobytes())
                response = json.loads(ws.recv())
                latest_faces.set(response.get('faces', []))
        except (websocket.WebSocketException, ConnectionError, OSError) as e:
            print(f"Conexão com o servidor perdida ({e}). Reconectando em {RECONNECT_DELAY_SECONDS}s...")
            time.sleep(RECONNECT_DELAY_SECONDS)
        finally:
            if ws is not None:
                ws.close()


def run():
    cap = create_camera()
    latest_frame = LatestValue(None)
    latest_faces = LatestValue([])
    stop_event = threading.Event()

    worker = threading.Thread(
        target=recognition_worker,
        args=(latest_frame, latest_faces, stop_event),
        daemon=True,
    )
    worker.start()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Falha ao capturar frame da câmera.")
                continue

            latest_frame.set(frame.copy())

            draw_faces(frame, latest_faces.get())
            cv2.imshow('Smart Glasses - Evaluation', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return
    finally:
        stop_event.set()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        run()
    except KeyboardInterrupt:
        pass
