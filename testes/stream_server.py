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

FRAME_INTERVAL_SECONDS = 1.0 / STREAM_FPS

camera = None


async def handle_client(websocket):
    peer = websocket.remote_address
    print(f"Cliente conectado: {peer}")
    try:
        while True:
            t0 = time.perf_counter()

            ret, frame = camera.read()
            if not ret:
                print("Falha ao capturar frame da câmera.")
                await asyncio.sleep(FRAME_INTERVAL_SECONDS)
                continue

            ok, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if not ok:
                continue

            frame_b64 = base64.b64encode(buffer).decode('ascii')
            await websocket.send(frame_b64)

            elapsed = time.perf_counter() - t0
            await asyncio.sleep(max(0.0, FRAME_INTERVAL_SECONDS - elapsed))
    except ConnectionClosed:
        pass
    finally:
        print(f"Cliente desconectado: {peer}")


async def main():
    global camera
    camera = create_camera()
    try:
        async with serve(handle_client, STREAM_WS_HOST, STREAM_WS_PORT) as server:
            print(f"Streaming de câmera em ws://{STREAM_WS_HOST}:{STREAM_WS_PORT}")
            print(f"FPS alvo: {STREAM_FPS}")
            await server.serve_forever()
    finally:
        camera.release()


def run():
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
