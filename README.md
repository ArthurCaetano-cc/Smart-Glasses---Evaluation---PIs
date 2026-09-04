# Smart-Glasses - Evaluation - PIs

Código que roda na Raspberry Pi. A única função da raspberry é capturar frames da câmera e
transmiti-los para o app Expo, que se conecta a ela via WebSocket -- todo o processamento
(detecção, reconhecimento, cadastro) acontece no servidor (`api/`), orquestrado pelo app.

## Uso

```bash
pip install -r requiriments.txt
python main.py
```

Configuração em `.env`:

- `STREAM_WS_HOST` / `STREAM_WS_PORT`: endereço em que o servidor de streaming escuta
  (padrão `0.0.0.0:8765`).
- `STREAM_FPS`: taxa de envio de frames.
- `CAMERA_INDEX`, `CAMERA_BACKEND`, `CAMERA_WIDTH`, `CAMERA_HEIGHT`: ver `testes/camera.py`
  (suporta Picamera2/libcamera para o módulo CSI, ou OpenCV/V4L2 para webcams USB).
- `JPEG_QUALITY`: qualidade de compressão dos frames enviados.

No app, informe `ws://<ip-da-raspberry>:8765` como host para conectar.
