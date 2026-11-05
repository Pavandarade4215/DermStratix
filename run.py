#!/usr/bin/env python3
import os, sys

current_dir = os.path.abspath(os.path.dirname(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from app import app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print('=' * 65)
    print(' DermStratix AI — Facial & Cutaneous Health Screening Platform')
    print('=' * 65)
    print(f' * Platform online at: http://127.0.0.1:{port}/')
    print('=' * 65)
    app.run(host='127.0.0.1', port=port, debug=False)
