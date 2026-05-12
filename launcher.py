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


if __name__ == "__main__":
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt")

    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = TeeStream(sys.stdout, log_file)
    sys.stderr = TeeStream(sys.stderr, log_file)

    print(f"Logging to {log_path}")

    app_path = os.path.join(os.path.dirname(__file__), "PySide2", "app.py")
    runpy.run_path(app_path, run_name="__main__")
