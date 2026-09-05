"""
Servidor de streaming (raspberry): a única função da raspberry nesta arquitetura é capturar
frames da câmera (via camera.py) e transmiti-los para o app Expo, que faz a conexão. A
raspberry não fala mais com o servidor de reconhecimento -- quem faz isso agora é o app.

Protocolo: WebSocket assíncrono (lib `websockets`). Para cada cliente conectado, um loop
captura um frame, codifica em JPEG, converte para base64 e envia como frame de texto --
texto (e não binário) para o app não precisar lidar com Blob/ArrayBuffer no React Native.
Pensado para um único viewer por vez; múltiplos clientes conectados fazem leituras
independentes da mesma câmera.
"""
import asyncio
import base64
import os
import time

import cv2
from dotenv import load_dotenv
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from camera import create_camera

load_dotenv()

STREAM_WS_HOST = os.getenv('STREAM_WS_HOST', '0.0.0.0')
STREAM_WS_PORT = int(os.getenv('STREAM_WS_PORT', '8765'))
STREAM_FPS = float(os.getenv('STREAM_FPS', '10'))
JPEG_QUALITY = int(os.getenv('JPEG_QUALITY', '80'))
IMAGE_FALLBACK_PATH = os.getenv('IMAGE_FALLBACK_PATH')
CAMERA_RECONNECT_INTERVAL = float(os.getenv('CAMERA_RECONNECT_INTERVAL', '5'))

FRAME_INTERVAL_SECONDS = 1.0 / STREAM_FPS

camera = None
fallback_frame_b64 = None
last_camera_attempt = 0.0


def load_fallback_frame():
    """Pré-carrega e codifica a imagem fixa usada quando a câmera falha."""
    if not IMAGE_FALLBACK_PATH:
        return None

    image = cv2.imread(IMAGE_FALLBACK_PATH)
    if image is None:
        print(f"Não foi possível carregar a imagem de fallback: {IMAGE_FALLBACK_PATH}")
        return None

    ok, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        print("Falha ao codificar a imagem de fallback.")
        return None

    print(f"Imagem de fallback carregada: {IMAGE_FALLBACK_PATH}")
    return base64.b64encode(buffer).decode('ascii')


def read_frame():
    """Lê um frame da câmera, tentando reconectar (com cooldown) quando ela está indisponível."""
    global camera, last_camera_attempt

    if camera is None:
        now = time.perf_counter()
        if now - last_camera_attempt < CAMERA_RECONNECT_INTERVAL:
            return False, None
        last_camera_attempt = now
        try:
            camera = create_camera()
            print("Câmera reconectada.")
        except Exception as e:
            print(f"Câmera indisponível ({e}); usando imagem de fallback.")
            return False, None

    try:
        ret, frame = camera.read()
    except Exception as e:
        print(f"Erro ao ler frame da câmera: {e}")
        ret, frame = False, None

    if not ret:
        try:
            camera.release()
        except Exception:
            pass
        camera = None
        return False, None

    return True, frame


async def handle_client(websocket):
    peer = websocket.remote_address
    print(f"Cliente conectado: {peer}")
    try:
        while True:
            t0 = time.perf_counter()

            ret, frame = read_frame()
            if ret:
                ok, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                frame_b64 = base64.b64encode(buffer).decode('ascii') if ok else None
            else:
                frame_b64 = fallback_frame_b64

            if frame_b64 is None:
                await asyncio.sleep(FRAME_INTERVAL_SECONDS)
                continue

            await websocket.send(frame_b64)

            elapsed = time.perf_counter() - t0
            await asyncio.sleep(max(0.0, FRAME_INTERVAL_SECONDS - elapsed))
    except ConnectionClosed:
        pass
    finally:
        print(f"Cliente desconectado: {peer}")


async def main():
    global camera, fallback_frame_b64
    fallback_frame_b64 = load_fallback_frame()
    try:
        camera = create_camera()
    except Exception as e:
        print(f"Câmera indisponível no início ({e}); iniciando com imagem de fallback.")
        camera = None
    try:
        async with serve(handle_client, STREAM_WS_HOST, STREAM_WS_PORT) as server:
            print(f"Streaming de câmera em ws://{STREAM_WS_HOST}:{STREAM_WS_PORT}")
            print(f"FPS alvo: {STREAM_FPS}")
            await server.serve_forever()
    finally:
        if camera is not None:
            camera.release()


def run():
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
