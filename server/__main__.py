#!/usr/bin/env python3
"""
Server module entry point

Allows running server as a module: python -m server
"""

import os

if __name__ == "__main__":
    from .app import app, PORT, HOST

    debug_mode = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'

    print(f"Starting server on {HOST or '127.0.0.1'}:{PORT}")
    print(f"  - API: http://{HOST or '127.0.0.1'}:{PORT}")
    print(f"  - Docs: http://{HOST or '127.0.0.1'}:{PORT}/docs")
    print(f"  - Dashboard: http://{HOST or '127.0.0.1'}:{PORT}/board")
    print(f"  - Debug mode: {debug_mode}")

    app.run(debug=debug_mode, port=PORT, host=HOST)
