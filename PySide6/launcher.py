import re
import sys
import os
import runpy
from datetime import datetime

_ANSI_ESCAPE = re.compile(r'\x1b\[[0-9;?]*[a-zA-Z]')


def _strip_ansi(text):
    return _ANSI_ESCAPE.sub('', text)


class TeeStream:
    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file
        self._buffer = ""

    def write(self, data):
        self.stream.write(data)
        # Split on newlines so we can process each line independently
        parts = data.split('\n')
        for i, part in enumerate(parts):
            if '\r' in part:
                # \r resets to the start of the line — discard buffered content
                # and take only whatever follows the last \r
                self._buffer = part.rsplit('\r', 1)[-1]
            else:
                self._buffer += part
            if i < len(parts) - 1:
                # There was a \n after this part — flush the line to the log
                clean = _strip_ansi(self._buffer)
                if clean.strip():
                    self.log_file.write(clean + '\n')
                    self.log_file.flush()
                self._buffer = ""

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

    def fileno(self):
        return self.stream.fileno()

    def isatty(self):
        return self.stream.isatty()


# napari draws through Qt's OpenGL. Under X11, Qt defaults to GLX, which segfaults
# napari on some Mesa drivers, while EGL works there. A segfault cannot be caught
# in-process, so a tiny napari viewer is first opened with EGL in a child process,
# and EGL is only used when that survives; otherwise Qt keeps its default.
_EGL_PROBE = (
    "import numpy as np, napari\n"
    "from qtpy.QtCore import QTimer\n"
    "v = napari.Viewer(show=True)\n"
    "v.add_image(np.zeros((32, 32), np.uint16))\n"
    "QTimer.singleShot(500, v.close)\n"
    "napari.run()\n"
)


def _choose_x11_opengl():
    if not sys.platform.startswith("linux") or "QT_XCB_GL_INTEGRATION" in os.environ:
        return
    platform = os.environ.get("QT_QPA_PLATFORM", "")
    if platform.startswith("wayland") or (not platform and os.environ.get("WAYLAND_DISPLAY")):
        return  # Qt runs on Wayland, which always draws through EGL
    import subprocess

    env = dict(os.environ, QT_XCB_GL_INTEGRATION="xcb_egl")
    try:
        works = subprocess.run(
            [sys.executable, "-c", _EGL_PROBE], env=env, capture_output=True, timeout=60
        ).returncode == 0
    except Exception:
        works = False
    if works:
        os.environ["QT_XCB_GL_INTEGRATION"] = "xcb_egl"
    print(f"[gl] OpenGL under X11: {'EGL' if works else 'GLX (EGL check failed)'}")


if __name__ == "__main__":
    _project_root = os.path.dirname(os.path.dirname(__file__))
    log_dir = os.path.join(_project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt")

    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = TeeStream(sys.stdout, log_file)
    sys.stderr = TeeStream(sys.stderr, log_file)

    # faulthandler  writes every thread's Python stack
    # straight into the log file, so the log shows where it crashed.
    import faulthandler
    faulthandler.enable(file=log_file, all_threads=True)

    print(f"Logging to {log_path}")
    _choose_x11_opengl()

    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    runpy.run_path(app_path, run_name="__main__")
