"""Container entrypoint: run the API (internal) and the Streamlit UI (public) together.

Python rather than a shell script, so Windows line endings can't break it.
If either process exits, the container exits and the platform restarts it.
"""
import os
import subprocess
import sys
import time

api = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "api:app", "--host", "127.0.0.1", "--port", "8000", "--workers", "1"]
)
ui = subprocess.Popen(
    [
        sys.executable, "-m", "streamlit", "run", "ui.py",
        "--server.port", os.getenv("PORT", "7860"),
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]
)
while api.poll() is None and ui.poll() is None:
    time.sleep(2)
for p in (api, ui):
    if p.poll() is None:
        p.terminate()
sys.exit(1)
