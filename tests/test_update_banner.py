"""Headless check of the update banner state machine and restart wiring."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import PROJECT_ROOT, require, sample_path  # noqa: E402

import os
import sys

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = PROJECT_ROOT
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "PySide6"))

from PySide6.QtWidgets import QApplication

import app as nuclogic

failures = []


def check(label, cond, detail=""):
    if not cond:
        failures.append(f"{label}: {detail}")
    print(f"  [{'PASS' if cond else 'FAIL'}] {label} {detail if not cond else ''}")


class Headless(nuclogic.MainWindow):
    """Skip the network check and the torch prewarm; keep all the UI."""

    def _start_update_check(self):
        pass

    def _start_prewarm(self):
        pass


qapp = QApplication.instance() or QApplication([])
w = Headless()
w.show()  # child isVisible() is only meaningful once the window is shown
qapp.processEvents()

print("\n== initial state ==")
check("banner hidden", w._update_banner.isVisible() is False)
check("restart button exists and hidden", w._restart_btn.isVisible() is False)

print("\n== update available ==")
w._on_update_available("0.0.5", "https://example/zip", "https://example/notes")
check("banner shown", w._update_banner.isVisible() is True)
check("install visible", w._install_btn.isVisible() is True)
check("install enabled", w._install_btn.isEnabled() is True)
check("release notes visible", w._release_notes_btn.isVisible() is True)
check("restart still hidden", w._restart_btn.isVisible() is False)

print("\n== update finished ==")
w._on_update_finished()
check("install button hidden", w._install_btn.isVisible() is False)
check("release notes hidden", w._release_notes_btn.isVisible() is False)
check("restart button visible", w._restart_btn.isVisible() is True)
check("restart button enabled", w._restart_btn.isEnabled() is True)
check("label mentions restart", "restart" in w._update_label.text().lower(),
      repr(w._update_label.text()))

print("\n== error path still re-enables install ==")
w2 = Headless()
w2.show()
w2._on_update_available("0.0.5", "u", "n")
w2._install_btn.setEnabled(False)
w2._on_update_error("network down")
check("install re-enabled on error", w2._install_btn.isEnabled() is True)
check("restart hidden on error", w2._restart_btn.isVisible() is False)
check("error text shown", "network down" in w2._update_label.text())

print("\n== restart blocked while segmentation runs ==")


class FakeWorker:
    def isRunning(self):
        return True


w3 = Headless()
w3.show()
w3._on_update_available("0.0.5", "u", "n")
w3._on_update_finished()
w3.worker = FakeWorker()
spawned = []
import subprocess

real_popen = subprocess.Popen
subprocess.Popen = lambda *a, **k: spawned.append((a, k))
try:
    w3._restart_app()
finally:
    subprocess.Popen = real_popen
check("no process spawned while busy", spawned == [], f"{spawned}")
check("restart button still enabled", w3._restart_btn.isEnabled() is True)
check("tells user to wait", "still running" in w3._update_label.text().lower(),
      repr(w3._update_label.text()))

print("\n== restart spawns the launcher and quits ==")
w4 = Headless()
w4.show()
w4._on_update_available("0.0.5", "u", "n")
w4._on_update_finished()
spawned = []
quit_called = []
subprocess.Popen = lambda *a, **k: spawned.append((a, k))
real_quit = QApplication.quit
QApplication.quit = staticmethod(lambda: quit_called.append(True))
try:
    w4._restart_app()
finally:
    subprocess.Popen = real_popen
    QApplication.quit = real_quit

check("one process spawned", len(spawned) == 1, f"{spawned}")
if spawned:
    command = spawned[0][0][0]
    kwargs = spawned[0][1]
    print(f"      command: {command}")
    expected = "NucLogic.bat" if sys.platform == "win32" else "Linux_NucLogic.sh"
    check("launches the launcher script",
          any(expected in str(part) for part in command), f"{command}")
    check("launcher path exists",
          any(os.path.exists(str(part)) for part in command if expected in str(part)))
    check("cwd is project root",
          os.path.normcase(kwargs.get("cwd", "")) == os.path.normcase(ROOT),
          f"{kwargs.get('cwd')}")
check("app quit called", quit_called == [True])

print("\n== spawn failure is reported, app stays open ==")
w5 = Headless()
w5.show()
w5._on_update_available("0.0.5", "u", "n")
w5._on_update_finished()
quit_called = []


def boom(*a, **k):
    raise OSError("access denied")


subprocess.Popen = boom
QApplication.quit = staticmethod(lambda: quit_called.append(True))
try:
    w5._restart_app()
finally:
    subprocess.Popen = real_popen
    QApplication.quit = real_quit

check("did not quit on failure", quit_called == [])
check("restart button re-enabled", w5._restart_btn.isEnabled() is True)
check("failure message shown", "access denied" in w5._update_label.text(),
      repr(w5._update_label.text()))

print("\n" + ("ALL PASS" if not failures else f"{len(failures)} FAILURES:"))
for f in failures:
    print("  -", f)
sys.exit(1 if failures else 0)
