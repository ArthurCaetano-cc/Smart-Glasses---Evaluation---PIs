"""
Cliente de identificação (raspberry): em loop contínuo, captura um frame da câmera,
envia para o endpoint HTTP /api/model/recognize da API (reconhecimento de rosto único --
ver api/routes/recognition.py) e exibe o resultado: a imagem e o nome da pessoa mais
parecida, a confiança e o status (autorizado / não autorizado / nenhum rosto detectado).

Diferente do main.py (que mostra o feed ao vivo com caixas para todos os rostos), aqui
não há preview da câmera nem overlay em tempo real -- só o resultado da última consulta.
"""
import os

import cv2
import numpy as np
import requests
from dotenv import load_dotenv

from camera import create_camera

load_dotenv()

SERVER_HTTP_URL = os.getenv('SERVER_HTTP_URL', 'http://localhost:5000')
JPEG_QUALITY = int(os.getenv('JPEG_QUALITY', '80'))
REQUEST_TIMEOUT_SECONDS = float(os.getenv('REQUEST_TIMEOUT_SECONDS', '10'))
RECONNECT_DELAY_SECONDS = 2
CAPTURE_INTERVAL_SECONDS = float(os.getenv('CAPTURE_INTERVAL_SECONDS', '0'))

WINDOW_NAME = 'Smart Glasses - Identificação'
PANEL_WIDTH = 420
PANEL_HEIGHT = 520
FACE_IMAGE_SIZE = 300
FACE_IMAGE_POS = (60, 80)

STATUS_STYLES = {
    'authorized': ('AUTORIZADO', (0, 200, 0)),
    'unauthorized': ('NAO AUTORIZADO', (0, 0, 220)),
    'no_face': ('NENHUM ROSTO DETECTADO', (110, 110, 110)),
    'error': ('SEM CONEXAO COM O SERVIDOR', (0, 140, 255)),
}


def query_server(jpeg_bytes):
    response = requests.post(
        f"{SERVER_HTTP_URL}/api/model/recognize",
        files={'image': ('frame.jpg', jpeg_bytes, 'image/jpeg')},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def fetch_matched_image(matched_image_path):
    if not matched_image_path:
        return None
    try:
        response = requests.get(
            f"{SERVER_HTTP_URL}/api/image",
            params={'path': matched_image_path},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
    except requests.RequestException as e:
        print(f"[WARNING] Não foi possível buscar a imagem do match: {e}")
        return None


def build_result_panel(status, name, confidence, face_image):
    label, color = STATUS_STYLES.get(status, ('ERRO', (0, 0, 0)))
    panel = np.full((PANEL_HEIGHT, PANEL_WIDTH, 3), 30, dtype=np.uint8)

    cv2.rectangle(panel, (0, 0), (PANEL_WIDTH, 60), color, -1)
    cv2.putText(panel, label, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    img_x, img_y = FACE_IMAGE_POS
    if face_image is not None:
        resized = cv2.resize(face_image, (FACE_IMAGE_SIZE, FACE_IMAGE_SIZE))
        panel[img_y:img_y + FACE_IMAGE_SIZE, img_x:img_x + FACE_IMAGE_SIZE] = resized
    else:
        cv2.rectangle(panel, (img_x, img_y), (img_x + FACE_IMAGE_SIZE, img_y + FACE_IMAGE_SIZE), (70, 70, 70), -1)
        cv2.putText(panel, "sem imagem", (img_x + 90, img_y + FACE_IMAGE_SIZE // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    text_y = img_y + FACE_IMAGE_SIZE + 50
    cv2.putText(panel, name or '--', (16, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(panel, f"Confianca: {confidence:.1f}%", (16, text_y + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    return panel


def run():
    cap = create_camera()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Falha ao capturar frame da câmera.")
                continue

            ok, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if not ok:
                continue

            try:
                result = query_server(buffer.tobytes())
            except requests.RequestException as e:
                print(f"Erro ao consultar o servidor ({e}). Tentando novamente em {RECONNECT_DELAY_SECONDS}s...")
                cv2.imshow(WINDOW_NAME, build_result_panel('error', None, 0.0, None))
                if cv2.waitKey(int(RECONNECT_DELAY_SECONDS * 1000)) & 0xFF == ord('q'):
                    return
                continue

            face_image = fetch_matched_image(result.get('matched_image'))
            panel = build_result_panel(
                result.get('status', 'no_face'),
                result.get('matched_name'),
                result.get('confidence', 0.0),
                face_image,
            )
            cv2.imshow(WINDOW_NAME, panel)

            wait_ms = max(1, int(CAPTURE_INTERVAL_SECONDS * 1000))
            if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
                return
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        run()
    except KeyboardInterrupt:
        pass
