"""
Entrypoint da raspberry: inicia o servidor de streaming de câmera (testes/stream_server.py)
que o app Expo consome diretamente. A raspberry não fala com o servidor de reconhecimento.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'testes'))

from stream_server import run

if __name__ == '__main__':
    run()
