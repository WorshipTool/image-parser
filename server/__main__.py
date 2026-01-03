#!/usr/bin/env python3
"""
Server module entry point

Allows running server as a module: python -m server
"""

if __name__ == "__main__":
    from .app import app, PORT, HOST

    print(f"Starting server on {HOST or '127.0.0.1'}:{PORT}")
    print(f"  - API: http://{HOST or '127.0.0.1'}:{PORT}")
    print(f"  - Docs: http://{HOST or '127.0.0.1'}:{PORT}/docs")
    print(f"  - Dashboard: http://{HOST or '127.0.0.1'}:{PORT}/board")

    app.run(debug=True, port=PORT, host=HOST)
