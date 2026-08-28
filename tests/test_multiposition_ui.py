"""The multiposition UI: scanning, the confirm gate, the worker, deletion.

Modal dialogs cannot be driven under the offscreen Qt platform, so
QMessageBox.question is replaced with a scripted answer. Everything else is the
real widget code.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import PROJECT_ROOT, Results  # noqa: E402

sys.path.insert(0, os.path.join(PROJECT_ROOT, "PySide6"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import tifffile  # noqa: E402

import app as nuclogic  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel  # noqa: E402

from utils.split_positions import MultipositionFile  # noqa: E402

r = Results()
qapp = QApplication.instance() or QApplication([])


class Headless(nuclogic.MainWindow):
    def _start_update_check(self):
        pass

    def _start_prewarm(self):
        pass


def answer_with(value):
    """Script the next QMessageBox.question answer."""
    nuclogic.QMessageBox.question = staticmethod(lambda *a, **k: value)


shown = []
nuclogic.QMessageBox.information = staticmethod(
    lambda parent, title, text, *a, **k: shown.append((title, text))
)
nuclogic.QMessageBox.critical = staticmethod(
    lambda parent, title, text, *a, **k: shown.append((title, text))
)
nuclogic.QMessageBox.warning = staticmethod(
    lambda parent, title, text, *a, **k: shown.append((title, text))
)

tmp = tempfile.mkdtemp()
REF = np.random.default_rng(1).integers(0, 4000, (2, 1, 3, 8, 8)).astype(np.uint16)


def labels_of(window):
    texts = []
    for layout in (window.load_data_layout_2, window.load_data_layout_2_1):
        for index in range(layout.count()):
            item = layout.itemAt(index)
            widget = item.widget()
            if isinstance(widget, QLabel):
                texts.append(widget.text())
            elif item.layout():
                for sub in range(item.layout().count()):
                    sub_widget = item.layout().itemAt(sub).widget()
                    if isinstance(sub_widget, QLabel):
                        texts.append(sub_widget.text())
    return texts


print("== the folder scan reaches the UI ==")
scan_dir = os.path.join(tmp, "scan")
os.makedirs(scan_dir)
tifffile.imwrite(os.path.join(scan_dir, "movie.nd2"), REF[0, 0])  # stand-in bytes
entry = MultipositionFile(os.path.join(scan_dir, "movie.nd2"), 46, [])
nuclogic.scan_directory = lambda directory: [entry]

window = Headless()
window.update_file_info(scan_dir)
texts = labels_of(window)
r.check("position count is shown before any click",
        any("46 stage positions" in t for t in texts), f"{texts}")

print("\n== an already-split file does not re-prompt ==")
split_entry = MultipositionFile(os.path.join(scan_dir, "movie.nd2"), 3,
                                ["movie_p01", "movie_p02", "movie_p03"])
nuclogic.scan_directory = lambda directory: [split_entry]
window.update_file_info(scan_dir)
texts = labels_of(window)
r.check("reports it as already split",
        any("already split" in t for t in texts), f"{texts}")
r.check("no 'create directories' prompt",
        not any("Create directories for them?" in t for t in texts), f"{texts}")

print("\n== declining the dialog splits nothing ==")
nuclogic.scan_directory = lambda directory: [entry]
window.update_file_info(scan_dir)
started = []
# MainWindow defines these itself, so the originals must be put back rather
# than deleted -- `del` would remove the real method, not just the override.
real_start_split = nuclogic.MainWindow._start_split
real_offer_deletion = nuclogic.MainWindow._offer_original_deletion
nuclogic.MainWindow._start_split = lambda self, d, p, nd=(): started.append(p)
answer_with(nuclogic.QMessageBox.No)
window._handle_create_dirs(scan_dir)
r.check("no split started when declined", started == [], f"{started}")

print("\n== accepting starts the split ==")
answer_with(nuclogic.QMessageBox.Yes)
window._handle_create_dirs(scan_dir)
r.check("split started when accepted", len(started) == 1, f"{started}")
nuclogic.MainWindow._start_split = real_start_split

print("\n== the worker reports progress and results ==")
work_dir = os.path.join(tmp, "work")
os.makedirs(work_dir)
source = os.path.join(work_dir, "movie.nd2")
open(source, "wb").close()

calls = {"progress": [], "status": [], "finished": []}


def fake_split(path, progress=None, should_stop=None):
    for step in range(1, 4):
        if should_stop and should_stop():
            return [f"{path}_p{i}" for i in range(1, step)]
        if progress:
            progress(step / 3, f"movie_p0{step}")
    return [f"{path}_p{i}" for i in range(1, 4)]


worker = nuclogic.SampleConversionWorker([(source, fake_split)])
worker.progress_updated.connect(calls["progress"].append)
worker.status_changed.connect(calls["status"].append)
worker.finished_converting.connect(calls["finished"].append)
worker.run()

r.check("progress climbs to 100", calls["progress"][-1] == 100, f"{calls['progress']}")
r.check("progress is monotonic",
        calls["progress"] == sorted(calls["progress"]), f"{calls['progress']}")
r.check("status names the position", calls["status"][-1] == "movie_p03",
        f"{calls['status']}")
r.check("finished carries the folders",
        len(calls["finished"]) == 1 and len(calls["finished"][0][source]) == 3,
        f"{calls['finished']}")
r.check("not marked cancelled", worker.was_cancelled() is False)

print("\n== cancelling stops it and reports partial work ==")
cancel_worker = nuclogic.SampleConversionWorker([(source, fake_split)])
cancel_worker.stop()
results = []
cancel_worker.finished_converting.connect(results.append)
cancel_worker.run()
r.check("cancelled before starting writes nothing", results == [{}], f"{results}")
r.check("marked cancelled", cancel_worker.was_cancelled() is True)

print("\n== worker errors surface instead of crashing ==")


def boom(path, progress=None, should_stop=None):
    raise RuntimeError("disk full")


errors = []
bad_worker = nuclogic.SampleConversionWorker([(source, boom)])
bad_worker.error_occurred.connect(errors.append)
finished = []
bad_worker.finished_converting.connect(finished.append)
bad_worker.run()
r.check("error is emitted", errors and "disk full" in errors[0], f"{errors}")
r.check("no finished signal after an error", finished == [], f"{finished}")

print("\n== deletion only offered for fully converted files ==")
complete_path = os.path.join(work_dir, "complete.nd2")
partial_path = os.path.join(work_dir, "partial.nd2")
open(complete_path, "wb").close()
open(partial_path, "wb").close()

complete = MultipositionFile(complete_path, 2, [])
partial = MultipositionFile(partial_path, 4, [])
written = {complete_path: ["a", "b"], partial_path: ["a"]}

answer_with(nuclogic.QMessageBox.Yes)
window._offer_original_deletion([complete, partial], written)
r.check("fully split original deleted", not os.path.exists(complete_path))
r.check("partially split original kept", os.path.exists(partial_path))

print("\n== declining deletion keeps everything ==")
kept_path = os.path.join(work_dir, "kept.nd2")
open(kept_path, "wb").close()
answer_with(nuclogic.QMessageBox.No)
window._offer_original_deletion([MultipositionFile(kept_path, 2, [])],
                                {kept_path: ["a", "b"]})
r.check("original kept when declined", os.path.exists(kept_path))

print("\n== a cancelled run never offers deletion ==")
offered = []
nuclogic.MainWindow._offer_original_deletion = lambda self, p, w: offered.append(p)


class CancelledWorker:
    def was_cancelled(self):
        return True


window.split_worker = CancelledWorker()
nuclogic.scan_directory = lambda directory: []
window._on_split_finished(scan_dir, [complete], {complete_path: ["a", "b"]})
r.check("no deletion prompt after cancel", offered == [], f"{offered}")
r.check("user is told it stopped",
        any("Conversion stopped" == title for title, _ in shown), f"{shown}")

nuclogic.MainWindow._offer_original_deletion = real_offer_deletion

print("\n== .nd acquisitions reach the same flow ==")


class FakeNd:
    def __init__(self, path, channels=("w1", "w2"), merged=False):
        self.path = path
        self.name = os.path.basename(path)
        self.channel_names = list(channels)
        self.already_merged = merged
        self.estimated_bytes = 1_000_000_000
        self.sources = [path] + [f"{path}_{c}.STK" for c in channels]


nd_dir = os.path.join(tmp, "nd")
os.makedirs(nd_dir)
nd_path = os.path.join(nd_dir, "acq.nd")
open(nd_path, "wb").close()

nuclogic.scan_directory = lambda directory: []
nuclogic.scan_nd_directory = lambda directory: [FakeNd(nd_path)]
window.update_file_info(nd_dir)
texts = labels_of(window)
r.check("nd acquisition is announced",
        any("MetaMorph acquisition" in t for t in texts), f"{texts}")

nuclogic.scan_nd_directory = lambda directory: [FakeNd(nd_path, merged=True)]
window.update_file_info(nd_dir)
texts = labels_of(window)
r.check("merged nd is not re-prompted",
        any("already been merged" in t for t in texts)
        and not any("Create directories for them?" in t for t in texts), f"{texts}")

print("\n== deleting a merged nd removes its stacks too ==")
sources = []
for suffix in (".nd", "_w1.STK", "_w2.STK"):
    p = os.path.join(nd_dir, "done" + suffix)
    open(p, "wb").close()
    sources.append(p)

entry = FakeNd(sources[0])
entry.sources = sources
answer_with(nuclogic.QMessageBox.Yes)
window._offer_original_deletion([entry], {sources[0]: ["folder"]})
r.check("nd index deleted", not os.path.exists(sources[0]))
r.check("stack files deleted too",
        not any(os.path.exists(p) for p in sources[1:]),
        [p for p in sources[1:] if os.path.exists(p)])

print("\n== a locked file does not take the app down ==")
# Reproduces the reported crash: a file still open elsewhere (the viewer holds
# lazily-loaded images open) cannot be renamed on Windows, and the unhandled
# PermissionError escaped the Qt slot.
lock_dir = os.path.join(tmp, "locked")
os.makedirs(lock_dir)
free_file = os.path.join(lock_dir, "free.tif")
held_file = os.path.join(lock_dir, "held.tif")
for path in (free_file, held_file):
    tifffile.imwrite(path, REF[0, 0])

nuclogic.scan_directory = lambda directory: []
nuclogic.scan_nd_directory = lambda directory: []
window.update_file_info(lock_dir)

handle = open(held_file, "rb")  # stands in for the viewer holding it open
try:
    from utils.file_to_folder import file_to_folder as real_file_to_folder

    failures = real_file_to_folder(lock_dir)
    if sys.platform == "win32":
        r.check("locked file is reported, not raised",
                [name for name, _ in failures] == ["held.tif"], f"{failures}")
        r.check("no empty folder left for it",
                not os.path.isdir(os.path.join(lock_dir, "held")))
    else:
        print("  (file locking is Windows-specific, skipped)", flush=True)
    r.check("the other file still got its folder",
            os.path.exists(os.path.join(lock_dir, "free", "free.tif")))

    shown.clear()
    window._handle_create_dirs(lock_dir)
    r.check("handler survives instead of crashing", True)
finally:
    handle.close()

print("\n== a partly merged nd keeps everything ==")
kept = []
for suffix in (".nd", "_w1.STK"):
    p = os.path.join(nd_dir, "partial" + suffix)
    open(p, "wb").close()
    kept.append(p)
partial_entry = FakeNd(kept[0])
partial_entry.sources = kept
window._offer_original_deletion([partial_entry], {kept[0]: []})
r.check("nothing deleted when no sample was written",
        all(os.path.exists(p) for p in kept))

r.finish()
