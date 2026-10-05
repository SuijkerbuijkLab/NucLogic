from datetime import datetime
import gc
import sys
import os
import re

# Pin the Qt binding for the whole process (app, napari, matplotlib-qtagg, and
# any qtpy-based widget). Without this, if anything imports qtpy before PySide6
# is loaded, qtpy can lock onto PyQt5 and napari then runs on a *different* Qt
# binding than this app -> two Qt bindings in one process -> hard crash.
os.environ.setdefault("QT_API", "pyside6")

import shutil
from pathlib import Path
if sys.platform == "win32":
    import winreg

import tifffile
import matplotlib
import hdf5plugin  # registers LZ4/shuffle HDF5 filters for all IMS reads in this process

matplotlib.use("qtagg")

parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
from utils.file_to_folder import file_to_folder
from utils.load_model import CELLPOSE_SAM_MODEL_NAME
from utils.load_image import SUPPORTED_EXTENSIONS, is_supported, load_image
from utils.split_positions import scan_directory, split_file
from utils.voxel_size import parse_override
from utils.merge_nd import merge_file, scan_directory as scan_nd_directory
from utils.update_checker import UpdateChecker, _read_local_version
from utils.updater import Updater
from utils.download_models import ModelDownloader, optional_assets
from utils.cropped_files import find_cropped_files, needs_crop, remove_stale_cropped_files

# Import PySide6 FIRST
from PySide6.QtCore import Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QIcon, QColor, QBrush
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QTabWidget,
    QFileDialog,
    QListWidget,
    QListWidgetItem,
    QAbstractItemView,
    QComboBox,
    QLineEdit,
    QGridLayout,
    QProgressBar,
    QMessageBox,
    QCheckBox,
    QSizePolicy,
    QScrollArea,
    QFrame,
    QDialog,
    QDialogButtonBox,
    QRadioButton,
    QButtonGroup,
)
import qdarktheme

# Measurement regions, mapping the tag used in column names to its UI label.
REGION_LABELS = {"nuclei": "Nuclei", "cytoplasm": "Cytoplasm", "whole_cell": "Whole cell"}

# A KeyError during a sample almost always means a column/channel lookup missed,
# which is what happens when channel names differ from the previous run.
CHANNEL_NAME_HINT = (
    "Hint: this looks like a missing name lookup — check that the channel names "
    "are spelled exactly as in previous runs for this sample."
)


def _is_key_error(error_line):
    """True if a formatted traceback's last line is a KeyError."""
    return error_line.lstrip().startswith("KeyError")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NucLogic")
        self.resize(1920, 1080)
        self.sample_list = None
        self.view_data_sample_list = None  # Separate list for View Data tab
        self.export_data_sample_list = None  # Separate list for Export Data tab
        self.samples_data = []  # Data structure holding samples
        self._set_icon()
        self._setup_ui()
        self._start_model_download()
        # GPU check is driven by _start_prewarm's background thread (which owns
        # the torch import) so it doesn't block the window from appearing.
        self._start_prewarm()

    def _set_icon(self):
        icon_path = Path(__file__).parent / "www" / "organoid_segmenter.ico"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

    # ── GPU warning ───────────────────────────────────────────────────────────

    def _create_gpu_warning_banner(self):
        self._gpu_banner = QFrame()
        self._gpu_banner.setVisible(False)
        self._gpu_banner.setStyleSheet(
            "QFrame { background: #8B5E00; border-radius: 0px; padding: 4px; }"
            "QLabel { color: white; background: transparent; }"
            "QPushButton { color: white; background: #6b4700; border: 1px solid #c48a00;"
            " border-radius: 4px; padding: 3px 10px; }"
            "QPushButton:hover { background: #7a5200; }"
        )
        row = QHBoxLayout(self._gpu_banner)
        row.setContentsMargins(12, 4, 12, 4)

        label = QLabel("⚠  No GPU found — segmentation will be extremely slow.")
        row.addWidget(label, 1)

        dismiss_btn = QPushButton("✕")
        dismiss_btn.setFixedWidth(28)
        dismiss_btn.clicked.connect(lambda: self._gpu_banner.setVisible(False))
        row.addWidget(dismiss_btn)

        return self._gpu_banner

    def _on_gpu_checked(self, available):
        # Driven by PrewarmWorker.gpu_available (runs on the UI thread via the
        # queued signal), so torch need not be imported before the window shows.
        if not available:
            print("[gpu] No CUDA-capable GPU found. Running in CPU mode.")
            self._gpu_banner.setVisible(True)

    # ── Background pre-warm ───────────────────────────────────────────────────

    def _start_prewarm(self):
        # Import heavy pipeline modules (incl. torch) in the background so the
        # window shows quickly and Run/View start instantly. Kept on self so the
        # thread isn't garbage-collected, and run at low priority so it never
        # competes with the UI.
        self._prewarm_worker = PrewarmWorker()
        self._prewarm_worker.gpu_available.connect(self._on_gpu_checked)
        self._prewarm_worker.start(QThread.Priority.LowPriority)

    # ── Update system ─────────────────────────────────────────────────────────

    def _create_update_banner(self):
        self._update_banner = QFrame()
        self._update_banner.setVisible(False)
        self._update_banner.setStyleSheet(
            "QFrame { background: #2a5298; border-radius: 0px; padding: 4px; }"
            "QLabel { color: white; background: transparent; }"
            "QPushButton { color: white; background: #1a3a78; border: 1px solid #4a72c8;"
            " border-radius: 4px; padding: 3px 10px; }"
            "QPushButton:hover { background: #1e4494; }"
        )
        row = QHBoxLayout(self._update_banner)
        row.setContentsMargins(12, 4, 12, 4)

        self._update_label = QLabel("")
        row.addWidget(self._update_label, 1)

        self._release_notes_btn = QPushButton("Release notes")
        self._release_notes_btn.setVisible(False)
        self._release_notes_btn.clicked.connect(self._open_release_notes)
        row.addWidget(self._release_notes_btn)

        self._install_btn = QPushButton("Install Update")
        self._install_btn.clicked.connect(self._install_update)
        row.addWidget(self._install_btn)

        self._restart_btn = QPushButton("Restart now")
        self._restart_btn.setVisible(False)
        self._restart_btn.clicked.connect(self._restart_app)
        row.addWidget(self._restart_btn)

        dismiss_btn = QPushButton("✕")
        dismiss_btn.setFixedWidth(28)
        dismiss_btn.clicked.connect(lambda: self._update_banner.setVisible(False))
        row.addWidget(dismiss_btn)

        return self._update_banner

    def _start_model_download(self):
        # Model weights ship as release assets, not in the repo, so a fresh clone
        # or zip arrives without them. Runs before the update check so the two
        # never compete for the banner.
        self._model_downloader = ModelDownloader()
        self._model_downloader.progress.connect(self._on_model_progress)
        self._model_downloader.completed.connect(self._on_models_ready)
        self._model_downloader.error.connect(self._on_model_error)
        self._model_downloader.start()

    def _on_model_progress(self, msg):
        self.run_segmentation_btn.setEnabled(False)
        self._install_btn.setVisible(False)
        self._update_label.setText(msg)
        self._update_banner.setVisible(True)

    def _on_models_ready(self):
        self.run_segmentation_btn.setEnabled(True)
        self._install_btn.setVisible(True)
        self._update_banner.setVisible(False)
        self._start_update_check()

    def _on_model_error(self, msg):
        print(f"[models] download failed: {msg}")
        self.run_segmentation_btn.setEnabled(False)
        self._install_btn.setVisible(False)
        self._update_label.setText(f"Could not download model weights: {msg}")
        self._update_banner.setVisible(True)

    def _start_update_check(self):
        # Print the installed version immediately (works offline); the async check
        # below then reports up-to-date / update-available once GitHub responds.
        current = _read_local_version()
        print(f"[version] NucLogic {current}" if current else "[version] NucLogic (version not yet set)")

        self._update_checker = UpdateChecker()
        self._update_checker.update_available.connect(self._on_update_available)
        self._update_checker.up_to_date.connect(
            lambda v: print(f"[update] NucLogic {v} is up to date.")
        )
        self._update_checker.check_failed.connect(
            lambda msg: print(f"[update] check failed: {msg}")
        )
        self._update_checker.start()

    def _on_update_available(self, version, url, html_url):
        self._update_url = url
        self._update_version = version
        current = _read_local_version() or "unknown"
        print(f"[update] NucLogic {version} is available (currently on: {current}).")
        self._update_label.setText(f"NucLogic {version} is available  (currently on: {current})")
        if html_url:
            self._release_notes_html_url = html_url
            self._release_notes_btn.setVisible(True)
        self._update_banner.setVisible(True)

    def _open_release_notes(self):
        url = getattr(self, "_release_notes_html_url", "")
        if url:
            QDesktopServices.openUrl(QUrl(url))

    def _install_update(self):
        self._install_btn.setEnabled(False)
        self._update_label.setText("Downloading…")
        self._updater = Updater(self._update_url, self._update_version)
        self._updater.progress.connect(self._update_label.setText)
        self._updater.finished.connect(self._on_update_finished)
        self._updater.error.connect(self._on_update_error)
        self._updater.start()

    def _on_update_finished(self):
        self._update_label.setText("Update complete — restart to apply it.")
        self._install_btn.setVisible(False)
        self._release_notes_btn.setVisible(False)
        self._restart_btn.setVisible(True)

    def _on_update_error(self, msg):
        self._update_label.setText(f"Update failed: {msg}")
        self._install_btn.setEnabled(True)

    def _restart_app(self):
        """Relaunch through the same launcher script, then close this instance.

        Going through NucLogic.bat / Linux_NucLogic.sh rather than re-executing
        the current interpreter matters after an update: the launcher re-enters
        the pixi environment, which may have just been rebuilt.
        """
        import subprocess

        if getattr(self, "worker", None) is not None and self.worker.isRunning():
            self._update_label.setText(
                "Segmentation is still running — restart once it has finished."
            )
            return

        self._restart_btn.setEnabled(False)
        project_root = os.path.dirname(os.path.dirname(__file__))
        fallback = [sys.executable, os.path.join(project_root, "PySide6", "launcher.py")]

        try:
            if sys.platform == "win32":
                launcher = os.path.join(project_root, "NucLogic.bat")
                command = (
                    ["cmd", "/c", "start", "", launcher]
                    if os.path.exists(launcher)
                    else fallback
                )
                subprocess.Popen(command, cwd=project_root)
            else:
                launcher = os.path.join(project_root, "Linux_NucLogic.sh")
                command = ["bash", launcher] if os.path.exists(launcher) else fallback
                subprocess.Popen(command, cwd=project_root, start_new_session=True)
        except Exception as e:
            self._restart_btn.setEnabled(True)
            self._update_label.setText(
                f"Could not restart automatically ({e}) — please close and reopen NucLogic."
            )
            return

        QApplication.quit()

    # ── End update system ─────────────────────────────────────────────────────

    def _setup_ui(self):
        tabs = QTabWidget()
        tabs.addTab(self._create_load_data_tab(), "  Load Data  ")
        tabs.addTab(self._create_segment_tab(), "  Segment  ")

        # Create view data tab
        self.view_data_widget = QWidget()
        self.view_data_layout = QVBoxLayout()
        self.view_data_layout.setSpacing(30)
        self.view_data_widget.setLayout(self.view_data_layout)
        tabs.addTab(self.view_data_widget, "  View Data  ")

        # Create export data tab
        self.export_data_widget = QWidget()
        self.export_data_layout = QVBoxLayout()
        self.export_data_layout.setAlignment(Qt.AlignTop)
        self.export_data_layout.setSpacing(30)
        self.export_data_widget.setLayout(self.export_data_layout)
        tabs.addTab(self.export_data_widget, "  Export Data  ")
        tabs.setStyleSheet("QTabBar::tab { padding: 10px 20px; }")

        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)
        container_layout.addWidget(self._create_gpu_warning_banner())
        container_layout.addWidget(self._create_update_banner())
        container_layout.addWidget(tabs)
        self.setCentralWidget(container)

    def _create_load_data_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(30)
        layout.setAlignment(Qt.AlignTop)

        # Path selection section
        layout.addLayout(self._create_path_layout())

        # File info section
        self.load_data_layout_2 = QVBoxLayout()
        self.load_data_layout_2.setSpacing(5)
        self.load_data_layout_2_1 = QHBoxLayout()
        layout.addLayout(self.load_data_layout_2)

        # Sample selection section
        self.load_data_layout_3 = QVBoxLayout()
        self.load_data_layout_3.setSpacing(5)
        layout.addLayout(self.load_data_layout_3, 1)

        widget.setLayout(layout)
        return widget

    def _create_path_layout(self):
        layout = QHBoxLayout()
        self.path_text = QLineEdit()
        self.path_text.setReadOnly(True)
        self.path_text.setPlaceholderText("No path selected")
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.browse_folder)
        layout.addWidget(self.path_text)
        layout.addWidget(browse_button)
        return layout

    def _clear_layout(self, layout):
        """Recursively clear all widgets from a layout"""
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
            elif child.layout():
                self._clear_layout(child.layout())
            del child

    def browse_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder", "")
        if folder_path:
            self.path_text.setText(folder_path)
            self.update_file_info(folder_path)

    def update_file_info(self, dir_name):
        self._clear_layout(self.load_data_layout_2)
        self._clear_layout(self.load_data_layout_3)
        self._clear_layout(self.view_data_layout)
        self._clear_layout(self.export_data_layout)
        self.sample_list = None
        self.view_data_sample_list = None
        self.export_data_sample_list = None

        if not Path(dir_name).exists():
            return

        # Store samples in data structure
        self.samples_data = self._get_samples(dir_name)
        image_files = self._get_image_files(dir_name)
        self.multiposition_files = self._scan_multiposition(dir_name)
        self.nd_files = self._scan_nd(dir_name)

        # Display samples count
        self.load_data_layout_2.addWidget(
            QLabel(f"Found {len(self.samples_data)} samples")
        )

        # Handle loose image files. A .nd is not an image file in its own right,
        # so it has to trigger this section too.
        if image_files or self.nd_files:
            self._add_loose_files_section(dir_name, image_files)

        # Create list widgets (only first time)
        if self.sample_list is None:
            self._create_sample_list_widgets()

        # Populate both lists with current samples data
        self._update_sample_lists()
        self._update_crop_status()

    def _create_sample_list_widgets(self):
        """Create the list widgets for Load Data and View Data tabs (first time only)"""
        # Load Data tab list
        self.sample_list = self._create_single_sample_list()
        self.load_data_layout_3.addWidget(self.sample_list)

    def _create_single_sample_list(self):
        """Create a single sample selection widget"""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(5)

        # Buttons
        button_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select all")
        deselect_all_btn = QPushButton("Deselect all")
        select_all_btn.clicked.connect(self.select_all_samples)
        deselect_all_btn.clicked.connect(self.deselect_all_samples)
        button_layout.addWidget(select_all_btn)
        button_layout.addWidget(deselect_all_btn)
        layout.addLayout(button_layout)

        # List widget
        sample_list = QListWidget()
        sample_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        sample_list.itemSelectionChanged.connect(self._update_segment_label)
        layout.addWidget(sample_list)

        widget.setLayout(layout)
        return widget

    def _get_samples(self, dir_name):
        """Get list of subdirectories (samples)"""
        return sorted(
            (d for d in os.listdir(dir_name) if os.path.isdir(os.path.join(dir_name, d))),
            key=lambda d: [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", d.lower())],
        )

    def _get_image_files(self, dir_name):
        """Get list of image files in root directory"""
        return [f for f in os.listdir(dir_name) if is_supported(f)]

    def _scan_multiposition(self, dir_name):
        """Loose files holding several stage positions. Reads headers only."""
        try:
            return scan_directory(dir_name)
        except Exception as error:
            print(f"Warning: could not scan for multiposition files: {error}")
            return []

    def _scan_nd(self, dir_name):
        """Loose MetaMorph .nd acquisitions. Reads the .nd text and one header."""
        try:
            return scan_nd_directory(dir_name)
        except Exception as error:
            print(f"Warning: could not scan for .nd acquisitions: {error}")
            return []

    def _add_loose_files_section(self, dir_name, image_files):
        """Add section for handling loose image files"""
        multiposition = getattr(self, "multiposition_files", [])
        nd_files = getattr(self, "nd_files", [])
        multiposition_names = {entry.name for entry in multiposition}
        ordinary = [f for f in image_files if f not in multiposition_names]
        pending = [entry for entry in multiposition if not entry.already_split]
        already_split = [entry for entry in multiposition if entry.already_split]
        pending_nd = [entry for entry in nd_files if not entry.already_merged]
        merged_nd = [entry for entry in nd_files if entry.already_merged]

        for entry in already_split:
            self.load_data_layout_2.addWidget(
                QLabel(
                    f"{entry.name} is already split into {entry.positions} "
                    f"position samples."
                )
            )
        for entry in merged_nd:
            self.load_data_layout_2.addWidget(
                QLabel(f"{entry.name} has already been merged into a sample.")
            )

        if not ordinary and not pending and not pending_nd:
            return

        message = []
        if ordinary:
            message.append(f"Found {len(ordinary)} image files without a directory.")
        for entry in pending:
            message.append(
                f"{entry.name} contains {entry.positions} stage positions."
            )
        for entry in pending_nd:
            message.append(
                f"{entry.name} is a MetaMorph acquisition split across "
                f"{len(entry.sources) - 1} stack files."
            )
        message.append("Create directories for them?")
        self.load_data_layout_2_1.addWidget(QLabel(" ".join(message)))

        create_button = QPushButton("Create directories")
        create_button.clicked.connect(lambda: self._handle_create_dirs(dir_name))
        self.load_data_layout_2_1.addWidget(create_button)
        self.load_data_layout_2.addLayout(self.load_data_layout_2_1)

    def _handle_create_dirs(self, dir_name):
        try:
            self._create_dirs(dir_name)
        except Exception as error:
            # A Qt slot must not let an exception escape: it would take the
            # whole application down rather than reporting the problem.
            QMessageBox.critical(
                self,
                "Could not create directories",
                f"{type(error).__name__}: {error}",
            )
            self.update_file_info(dir_name)

    def _create_dirs(self, dir_name):
        pending = [
            entry
            for entry in getattr(self, "multiposition_files", [])
            if not entry.already_split
        ]
        pending_nd = [
            entry for entry in getattr(self, "nd_files", []) if not entry.already_merged
        ]

        # Release any lazily-loaded array nothing references any more: while one
        # is alive its source file stays open, and an open file cannot be moved.
        gc.collect()

        # Multiposition files are left alone by file_to_folder, and .nd/.STK are
        # not image formats it recognises, so this only deals with the ordinary
        # loose files either way. The scan already knows the position counts, so
        # pass them rather than re-opening every file.
        failures = file_to_folder(
            dir_name,
            {entry.name: entry.positions
             for entry in getattr(self, "multiposition_files", [])},
        )
        if failures:
            self._report_move_failures(failures)

        if (pending or pending_nd) and self._confirm_conversion(pending, pending_nd):
            self._start_split(dir_name, pending, pending_nd)
            return

        self.update_file_info(dir_name)

    def _report_move_failures(self, failures):
        """Tell the user which files stayed put, and why.

        On Windows a file cannot be moved while anything still has it open, and
        the usual culprit is the sample still being displayed in the viewer.
        """
        listed = "\n".join(f"    {name}" for name, _ in failures)
        QMessageBox.warning(
            self,
            "Some files could not be moved",
            f"{len(failures)} file(s) are open in another program and were left "
            f"where they are:\n\n{listed}\n\n"
            f"If one of them is still open in the napari viewer, close that "
            f"window and press Create directories again. Every other file was "
            f"moved into its own folder.",
        )

    def _confirm_conversion(self, pending, pending_nd):
        """Ask before spending the disk and time a conversion costs."""
        estimate_gb = sum(
            entry.estimated_bytes for entry in list(pending) + list(pending_nd)
        ) / 1e9

        lines = []
        if pending:
            positions = sum(entry.positions for entry in pending)
            lines.append(
                f"{len(pending)} file(s) contain more than one stage position. "
                f"Each position becomes its own sample ({positions} in total). "
                f"Without splitting, only the first position of each is analysed."
            )
            lines += [
                f"    {entry.name} — {entry.positions} positions" for entry in pending
            ]
        if pending_nd:
            if lines:
                lines.append("")
            lines.append(
                f"{len(pending_nd)} MetaMorph acquisition(s) are split across "
                f"separate stack files. Each becomes one sample with its "
                f"wavelengths merged into channels."
            )
            lines += [
                f"    {entry.name} — {len(entry.channel_names)} channels "
                f"({', '.join(entry.channel_names)})"
                for entry in pending_nd
            ]

        answer = QMessageBox.question(
            self,
            "Convert files into samples?",
            "\n".join(lines)
            + f"\n\nThis writes roughly {estimate_gb:.1f} GB of new files. "
            f"The originals are kept, and you will be asked afterwards whether "
            f"to delete them.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        return answer == QMessageBox.Yes

    def _start_split(self, dir_name, pending, pending_nd=()):
        self._clear_layout(self.load_data_layout_2_1)

        self.split_label = QLabel("Preparing…")
        self.split_progress = QProgressBar()
        self.split_progress.setRange(0, 100)
        self.split_cancel_btn = QPushButton("Cancel")
        self.split_cancel_btn.clicked.connect(self._cancel_split)

        self.load_data_layout_2_1.addWidget(self.split_label)
        self.load_data_layout_2_1.addWidget(self.split_progress)
        self.load_data_layout_2_1.addWidget(self.split_cancel_btn)
        self.load_data_layout_2.addLayout(self.load_data_layout_2_1)

        jobs = [(entry.path, split_file) for entry in pending]
        # merge_file returns one folder; the worker collects lists either way.
        jobs += [
            (entry.path,
             lambda path, progress, should_stop:
                 [f for f in [merge_file(path, progress, should_stop)] if f])
            for entry in pending_nd
        ]

        self.split_worker = SampleConversionWorker(jobs)
        self.split_worker.progress_updated.connect(self.split_progress.setValue)
        self.split_worker.status_changed.connect(self.split_label.setText)
        self.split_worker.error_occurred.connect(
            lambda message: self._on_split_error(dir_name, message)
        )
        self.split_worker.finished_converting.connect(
            lambda written: self._on_split_finished(
                dir_name, list(pending) + list(pending_nd), written
            )
        )
        self.split_worker.start()

    def _cancel_split(self):
        if getattr(self, "split_worker", None):
            self.split_worker.stop()
        self.split_cancel_btn.setEnabled(False)
        self.split_label.setText("Finishing the current position…")

    def _on_split_error(self, dir_name, message):
        QMessageBox.critical(self, "Splitting failed", message)
        self.update_file_info(dir_name)

    def _on_split_finished(self, dir_name, pending, written):
        cancelled = self.split_worker.was_cancelled()

        if cancelled:
            done = sum(len(folders) for folders in written.values())
            QMessageBox.information(
                self,
                "Conversion stopped",
                f"Stopped after writing {done} sample(s). What was already "
                f"written is usable; run this again to finish the rest.",
            )
        else:
            self._offer_original_deletion(pending, written)

        self.update_file_info(dir_name)

    def _offer_original_deletion(self, pending, written):
        """Offer to delete sources, but only ones that fully converted.

        Deleting acquisition data cannot be undone, so a file qualifies only
        once every sample it should produce has been written and re-read. For a
        .nd that means deleting its stack files too, not just the index.
        """
        complete = [
            entry
            for entry in pending
            if len(written.get(entry.path, [])) == self._expected_samples(entry)
        ]
        if not complete:
            return

        # A .nd lists its stacks; anything else is a single file.
        removable = {
            entry: list(getattr(entry, "sources", [entry.path])) for entry in complete
        }
        names = "\n".join(
            f"    {entry.name}"
            + (f" (and {len(files) - 1} stack files)" if len(files) > 1 else "")
            for entry, files in removable.items()
        )
        answer = QMessageBox.question(
            self,
            "Delete the original files?",
            f"Wrote {sum(len(written[e.path]) for e in complete)} sample(s) from "
            f"{len(complete)} acquisition(s), and every one reads back "
            f"correctly.\n\n{names}\n\n"
            f"Delete the original file(s)? This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return

        failed = []
        for entry, files in removable.items():
            for path in files:
                try:
                    os.remove(path)
                except OSError as error:
                    failed.append(f"{os.path.basename(path)}: {error}")
        if failed:
            QMessageBox.warning(
                self, "Could not delete", "\n".join(failed)
            )

    @staticmethod
    def _expected_samples(entry):
        """How many sample folders a source should produce when it converts."""
        return getattr(entry, "positions", 1)

    def _update_sample_lists(self):
        """Update both sample lists with data from self.samples_data"""
        # Get the list widget from Load Data tab
        load_data_list = self._get_list_widget(self.sample_list)
        load_data_list.clear()
        load_data_list.addItems(self.samples_data)

        # Create/update View Data tab list (only first time)
        if self.view_data_sample_list is None:
            self._setup_view_data_widgets()

        # Populate View Data list
        view_data_list = self._get_list_widget(self.view_data_sample_list)
        view_data_list.clear()
        view_data_list.addItems(self.samples_data)

        # Create/update Export Data tab list (only first time)
        if self.export_data_sample_list is None:
            self._setup_export_data_widgets()

        # Populate Export Data list
        export_data_list = self._get_list_widget(self.export_data_sample_list)
        export_data_list.clear()
        export_data_list.addItems(self.samples_data)

        # Sync selections between lists
        self._sync_selections()

    def _get_list_widget(self, parent_widget):
        """Extract the QListWidget from a parent widget"""
        layout = parent_widget.layout()
        for i in range(layout.count()):
            widget = layout.itemAt(i).widget()
            if isinstance(widget, QListWidget):
                return widget
        return None

    def _sync_selections(self):
        """Keep selections in sync between all lists"""
        load_data_list = self._get_list_widget(self.sample_list)

        for target_list_widget in [
            self.view_data_sample_list,
            self.export_data_sample_list,
        ]:
            if target_list_widget is None:
                continue
            target_list = self._get_list_widget(target_list_widget)
            target_list.blockSignals(True)
            target_list.clearSelection()
            for item in load_data_list.selectedItems():
                matching_items = target_list.findItems(item.text(), Qt.MatchExactly)
                if matching_items:
                    matching_items[0].setSelected(True)
            target_list.blockSignals(False)

    def select_all_samples(self):
        self._get_list_widget(self.sample_list).selectAll()
        self._sync_selections()

    def deselect_all_samples(self):
        self._get_list_widget(self.sample_list).clearSelection()
        self._sync_selections()

    def get_selected_samples(self):
        """Get list of selected sample names"""
        list_widget = self._get_list_widget(self.sample_list)
        return [item.text() for item in list_widget.selectedItems()]

    def _make_separator(self):
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        return line

    def _info_label(self, tooltip_text):
        lbl = QLabel("ⓘ")
        lbl.setToolTip(tooltip_text)
        lbl.setStyleSheet("color: #5599CC; font-size: 14px;")
        lbl.setCursor(Qt.PointingHandCursor)
        return lbl

    def _create_segment_tab(self):
        outer_widget = QWidget()
        outer_layout = QVBoxLayout()
        outer_layout.setSpacing(0)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        # Scrollable settings area
        settings_widget = QWidget()
        settings_layout = QVBoxLayout()
        settings_layout.setAlignment(Qt.AlignTop)
        settings_layout.setSpacing(30)
        settings_layout.setContentsMargins(9, 9, 9, 9)
        settings_layout.addLayout(self._create_channel_config_layout())
        settings_layout.addWidget(self._make_separator())
        settings_layout.addLayout(self._create_cropping_layout())
        settings_layout.addWidget(self._make_separator())
        settings_layout.addLayout(self._create_phenotype_settings_layout())
        settings_layout.addWidget(self._make_separator())
        settings_layout.addLayout(self._create_advanced_statistics_layout())
        settings_layout.addWidget(self._make_separator())
        settings_layout.addLayout(self._create_advanced_settings_layout())
        settings_widget.setLayout(settings_layout)

        scroll = QScrollArea()
        scroll.setWidget(settings_widget)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer_layout.addWidget(scroll, 1)

        # Fixed bottom strip — always visible, never scrolled away
        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout()
        bottom_layout.setContentsMargins(9, 6, 9, 9)
        bottom_layout.setSpacing(5)
        bottom_layout.addLayout(self._create_run_mode_layout())
        bottom_layout.addLayout(self._create_run_segmentation_layout())
        bottom_layout.addLayout(self._create_progress_bar())
        bottom_widget.setLayout(bottom_layout)
        outer_layout.addWidget(bottom_widget)

        outer_widget.setLayout(outer_layout)
        return outer_widget

    def _create_run_mode_layout(self):
        layout = QHBoxLayout()
        layout.addWidget(QLabel("Run mode:"))
        self.run_mode_combo = QComboBox()
        self.run_mode_combo.addItems([
            "Full pipeline",
            "Phenotype/cell-type calling only",
            "Statistics only",
            "Statistics + phenotype/cell-type calling",
        ])
        layout.addWidget(self.run_mode_combo)
        layout.addWidget(self._info_label(
            "Controls which pipeline steps run — non-full-pipeline modes skip cropping & segmentation and only work on already-segmented samples"
        ))
        layout.addStretch()
        return layout

    def _create_advanced_statistics_layout(self):
        layout = QVBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignTop)
        self.adv_stats_btn = QPushButton("Show advanced statistics settings")
        self.adv_stats_btn.setCheckable(True)
        self.adv_stats_btn.toggled.connect(self._toggle_advanced_statistics)
        self.adv_stats_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        layout.addWidget(self.adv_stats_btn)

        # Create advanced statistics settings widget
        self.advanced_statistics_widget = QWidget()
        self.advanced_statistics_widget.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum
        )
        self.advanced_statistics_widget.setLayout(self._create_advanced_statistics())
        self.advanced_statistics_widget.setVisible(False)
        layout.addWidget(self.advanced_statistics_widget)
        return layout

    def _toggle_advanced_statistics(self, checked):
        self.advanced_statistics_widget.setVisible(checked)
        self.adv_stats_btn.setText(
            "Hide advanced statistics settings" if checked else "Show advanced statistics settings"
        )

    def _create_advanced_statistics(self):
        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignTop)

        # layout.addWidget(
        #     QLabel(
        #         "Calculate extra statistics (Centroid, Bounding box, Volume, and Mean intensities are calculated by default)."
        #     )
        # )
        layout1 = QHBoxLayout()
        layout1.addWidget(QLabel("Measure intensities inside the:"))
        # Nuclei are always measured; the expanded regions are opt-in and can be
        # combined, so each run can report all three.
        self.region_checkboxes = {}
        for tag, label in REGION_LABELS.items():
            box = QCheckBox(label)
            if tag == "nuclei":
                box.setChecked(True)
                box.setEnabled(False)
            else:
                box.toggled.connect(self._region_selection_changed)
            self.region_checkboxes[tag] = box
            layout1.addWidget(box)
        layout1.addStretch()
        layout.addLayout(layout1)

        measurement_sub_layout = QVBoxLayout()
        measurement_sub_layout.setSpacing(6)
        measurement_sub_layout.setContentsMargins(30, 0, 0, 0)

        layout_cytoplasm = QHBoxLayout()
        self.QLabel_cytoplasm_size = QLabel("Extension radius (µm):")
        self.QLabel_cytoplasm_size.setVisible(False)
        layout_cytoplasm.addWidget(self.QLabel_cytoplasm_size)
        self.cytoplasm_size_input = QLineEdit("5")
        self.cytoplasm_size_input.setMaximumWidth(90)
        self.cytoplasm_size_input.setVisible(False)
        layout_cytoplasm.addWidget(self.cytoplasm_size_input)
        layout_cytoplasm.addStretch()
        measurement_sub_layout.addLayout(layout_cytoplasm)

        layout_mask = QHBoxLayout()
        self.save_measurement_mask_checkbox = QCheckBox(
            "Also save generated cytoplasm/whole-cell segmentation mask"
        )
        self.save_measurement_mask_checkbox.setVisible(False)
        self.save_measurement_mask_checkbox.setChecked(False)
        layout_mask.addWidget(self.save_measurement_mask_checkbox)
        layout_mask.addStretch()
        measurement_sub_layout.addLayout(layout_mask)

        layout.addLayout(measurement_sub_layout)

        layout2 = QHBoxLayout()
        layout2.addWidget(QLabel("Additional properties to calculate (multi-select):"))
        layout2.addWidget(self._info_label(
            "Additional properties from skimage.measure.regionprops.\n"
            "More info: https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops"
        ))
        self.calculate_advanced_statistics_list = QListWidget()
        self.calculate_advanced_statistics_list.setSelectionMode(
            QAbstractItemView.SelectionMode.MultiSelection
        )
        self.calculate_advanced_statistics_list.setMaximumHeight(180)

        # NOTE: several entries (anisotropy, axis_medial_length, elongation,
        # flatness, sphericity) are derived, voxel-size-scaled shape descriptors
        # computed in utils/compute_shape_descriptors.py, not plain regionprops.
        advanced_property_items = [
            "anisotropy",
            "area_bbox",
            "area_convex",
            "area_filled",
            "axis_major_length",
            "axis_medial_length",
            "axis_minor_length",
            "centroid_local",
            "centroid_weighted",
            "centroid_weighted_local",
            "coords",
            "coords_scaled",
            "elongation",
            "equivalent_diameter_area",
            "euler_number",
            "extent",
            "feret_diameter_max",
            "flatness",
            "image",
            "image_convex",
            "image_filled",
            "image_intensity",
            "inertia_tensor",
            "inertia_tensor_eigvals",
            "intensity_max",
            "intensity_mean",
            "intensity_min",
            "intensity_std",
            "moments",
            "moments_central",
            "moments_normalized",
            "moments_weighted",
            "moments_weighted_central",
            "moments_weighted_normalized",
            "num_pixels",
            "slice",
            "solidity",
            "sphericity",
        ]

        for item in advanced_property_items:
            self.calculate_advanced_statistics_list.addItem(QListWidgetItem(item))

        layout2.addWidget(self.calculate_advanced_statistics_list)
        layout2.addStretch()
        layout.addLayout(layout2)

        self.calculate_neighbour_statistics_checkbox = QCheckBox(
            "Calculate neighbour statistics"
        )
        self.calculate_neighbour_statistics_checkbox.setChecked(False)
        self.calculate_neighbour_statistics_checkbox.toggled.connect(
            self._toggle_neighbour_statistics
        )
        layout.addWidget(self.calculate_neighbour_statistics_checkbox)

        neighbour_sub_layout = QVBoxLayout()
        neighbour_sub_layout.setSpacing(6)
        neighbour_sub_layout.setContentsMargins(30, 0, 0, 0)

        self.calculate_neighbours_knn_checkbox = QCheckBox(
            "Calculate neighbours using KNN"
        )
        self.calculate_neighbours_knn_checkbox.setChecked(False)
        self.calculate_neighbours_knn_checkbox.toggled.connect(
            self._toggle_knn_neighbour_options
        )
        self.knn_checkbox_row_widget = QWidget()
        knn_row = QHBoxLayout(self.knn_checkbox_row_widget)
        knn_row.setContentsMargins(0, 0, 0, 0)
        knn_row.addWidget(self.calculate_neighbours_knn_checkbox)
        knn_row.addWidget(self._info_label(
            "K-Nearest Neighbours (KNN): finds the K cells whose centroids are closest in 3D space and treats them as neighbours.\n"
            "This is a purely distance-based approach — cells do not need to be physically touching.\n"
            "If you want to compute statistics for multiple K values, input them separated by commas (e.g. 3,5,7)"
        ))
        knn_row.addStretch()
        neighbour_sub_layout.addWidget(self.knn_checkbox_row_widget)

        self.knn_neighbour_options_widget = QWidget()
        knn_neighbour_options_layout = QVBoxLayout(self.knn_neighbour_options_widget)
        knn_neighbour_options_layout.setContentsMargins(30, 0, 0, 0)
        knn_neighbour_options_layout.setSpacing(6)

        layout_knn = QHBoxLayout()
        layout_knn.addWidget(QLabel("Number of neareist neighbours (K):"))
        self.knn_input = QLineEdit()
        self.knn_input.setPlaceholderText("e.g. 5 or 3,5,7")
        self.knn_input.setMaximumWidth(200)
        layout_knn.addWidget(self.knn_input)
        layout_knn.addStretch()
        knn_neighbour_options_layout.addLayout(layout_knn)

        self.calculate_phenotype_similarity_knn_checkbox = QCheckBox(
            "Calculate phenotype similarity score based on KNN"
        )
        self.calculate_phenotype_similarity_knn_checkbox.setChecked(False)
        knn_sim_row = QHBoxLayout()
        knn_sim_row.addWidget(self.calculate_phenotype_similarity_knn_checkbox)
        knn_sim_row.addWidget(self._info_label(
            "Calculates what fraction of a cell's K nearest neighbours share the same phenotype/cell-type. "
            "A score of 1.0 means all neighbours are the same type; 0.0 means none are."
        ))
        knn_sim_row.addStretch()
        knn_neighbour_options_layout.addLayout(knn_sim_row)
        neighbour_sub_layout.addWidget(self.knn_neighbour_options_widget)

        self.calculate_neighbours_touching_checkbox = QCheckBox(
            "Calculate neighbours that touch in 3D"
        )
        self.calculate_neighbours_touching_checkbox.setChecked(False)
        self.calculate_neighbours_touching_checkbox.toggled.connect(
            self._toggle_touching_neighbour_options
        )
        self.touching_checkbox_row_widget = QWidget()
        touching_row = QHBoxLayout(self.touching_checkbox_row_widget)
        touching_row.setContentsMargins(0, 0, 0, 0)
        touching_row.addWidget(self.calculate_neighbours_touching_checkbox)
        touching_row.addWidget(self._info_label(
            "Dilates each nucleus mask outward by a given radius (in µm) using a Euclidean Distance Transform (EDT), "
            "then checks which other nuclei overlap with the expanded region.\n"
            "Cells that overlap are considered neighbours — mimicking physical contact between cells."
        ))
        touching_row.addStretch()
        neighbour_sub_layout.addWidget(self.touching_checkbox_row_widget)

        self.touching_neighbour_options_widget = QWidget()
        touching_options_layout = QVBoxLayout(self.touching_neighbour_options_widget)
        touching_options_layout.setContentsMargins(30, 0, 0, 0)
        touching_options_layout.setSpacing(6)

        layout_touching_dilation = QHBoxLayout()
        layout_touching_dilation.addWidget(QLabel("Dilation radius (µm):"))
        self.touching_dilation_um_input = QLineEdit("1.0")
        self.touching_dilation_um_input.setMaximumWidth(120)
        layout_touching_dilation.addWidget(self.touching_dilation_um_input)
        layout_touching_dilation.addStretch()
        touching_options_layout.addLayout(layout_touching_dilation)

        self.calculate_phenotype_similarity_touching_checkbox = QCheckBox(
            "Calculate phenotype similarity score based on 3D touching"
        )
        self.calculate_phenotype_similarity_touching_checkbox.setChecked(False)
        touching_sim_row = QHBoxLayout()
        touching_sim_row.addWidget(self.calculate_phenotype_similarity_touching_checkbox)
        touching_sim_row.addWidget(self._info_label(
            "Calculates what fraction of a cell's touching neighbours share the same phenotype/cell-type.\n"
            "A score of 1.0 means all touching neighbours are the same type; 0.0 means none are."
        ))
        touching_sim_row.addStretch()
        touching_options_layout.addLayout(touching_sim_row)
        neighbour_sub_layout.addWidget(self.touching_neighbour_options_widget)
        layout.addLayout(neighbour_sub_layout)

        self._toggle_neighbour_statistics(
            self.calculate_neighbour_statistics_checkbox.isChecked()
        )
        self._toggle_knn_neighbour_options(
            self.calculate_neighbours_knn_checkbox.isChecked()
        )
        self._toggle_touching_neighbour_options(
            self.calculate_neighbours_touching_checkbox.isChecked()
        )

        return layout

    def _select_advanced_statistic(self, name):
        """Select one entry in the advanced-statistics list and scroll to it."""
        for i in range(self.calculate_advanced_statistics_list.count()):
            item = self.calculate_advanced_statistics_list.item(i)
            if item.text() == name:
                if not item.isSelected():
                    item.setSelected(True)
                    self.calculate_advanced_statistics_list.scrollToItem(item)
                return

    def _region_combo(self):
        """Region picker holding the column tag as item data."""
        combo = QComboBox()
        for tag, label in REGION_LABELS.items():
            combo.addItem(label, tag)
        combo.currentTextChanged.connect(self._region_selection_changed)
        return combo

    def get_measure_regions(self):
        """Region tags ticked in the advanced statistics settings."""
        return [tag for tag, box in self.region_checkboxes.items() if box.isChecked()]

    def phenotype_regions(self):
        """Regions the two phenotype channels are measured in."""
        return {
            self.phenotype_1_region.currentData(),
            self.phenotype_2_region.currentData(),
        }

    def _region_selection_changed(self):
        """Show the extension settings whenever an expanded region is wanted —
        by the statistics checkboxes or by either phenotype channel."""
        regions = set(self.get_measure_regions()) | self.phenotype_regions()
        expanded = bool(regions - {"nuclei", None})
        self.QLabel_cytoplasm_size.setVisible(expanded)
        self.cytoplasm_size_input.setVisible(expanded)
        self.save_measurement_mask_checkbox.setVisible(expanded)
        if expanded:
            # An expanded region is pointless without a mean intensity, so select it
            # rather than making the user find it in the advanced settings.
            self._select_advanced_statistic("intensity_mean")
        else:
            self.save_measurement_mask_checkbox.setChecked(False)

    def _toggle_neighbour_statistics(self, checked):
        self.knn_checkbox_row_widget.setVisible(checked)
        self.touching_checkbox_row_widget.setVisible(checked)
        self.knn_neighbour_options_widget.setVisible(
            checked and self.calculate_neighbours_knn_checkbox.isChecked()
        )
        self.touching_neighbour_options_widget.setVisible(
            checked and self.calculate_neighbours_touching_checkbox.isChecked()
        )

    def _toggle_knn_neighbour_options(self, checked):
        self.knn_neighbour_options_widget.setVisible(
            checked and self.calculate_neighbour_statistics_checkbox.isChecked()
        )

    def _toggle_touching_neighbour_options(self, checked):
        self.touching_neighbour_options_widget.setVisible(
            checked and self.calculate_neighbour_statistics_checkbox.isChecked()
        )

    def get_selected_advanced_statistics(self):
        return [
            item.text()
            for item in self.calculate_advanced_statistics_list.selectedItems()
        ]

    def get_advanced_statistics_settings(self):
        measure_regions = self.get_measure_regions()
        cytoplasm_size = 5
        # The radius is only needed once an expanded region is measured somewhere.
        if (set(measure_regions) | self.phenotype_regions()) - {"nuclei"}:
            try:
                cytoplasm_size = int(self.cytoplasm_size_input.text())
            except ValueError as exc:
                raise ValueError(
                    "Invalid cytoplasm size. Please enter a positive integer."
                ) from exc
            if cytoplasm_size < 1:
                raise ValueError(
                    "Invalid cytoplasm size. Please enter a positive integer."
                )

        extra_props = self.get_selected_advanced_statistics()
        run_mode = self.run_mode_combo.currentText()
        advanced_statistics_only = run_mode in ("Statistics only", "Statistics + phenotype/cell-type calling")
        save_measurement_mask = self.save_measurement_mask_checkbox.isChecked()
        calculate_neighbour_statistics = (
            self.calculate_neighbour_statistics_checkbox.isChecked()
        )
        use_knn_neighbours = False
        calculate_phenotype_similarity_knn = False
        knn_list = None
        use_touching_neighbours_3d = False
        touching_dilation_um = None
        calculate_phenotype_similarity_touching = False

        if calculate_neighbour_statistics:
            use_knn_neighbours = self.calculate_neighbours_knn_checkbox.isChecked()
            use_touching_neighbours_3d = (
                self.calculate_neighbours_touching_checkbox.isChecked()
            )

            if not use_knn_neighbours and not use_touching_neighbours_3d:
                raise ValueError(
                    "Please select at least one neighbour method (KNN and/or 3D touching)."
                )

            if use_knn_neighbours:
                knn_text = self.knn_input.text().strip()
                if knn_text == "":
                    raise ValueError(
                        "KNN is enabled. Please enter one or more K values (e.g. 5 or 3,5,7)."
                    )
                try:
                    # make list of ints if multiple K values entered, otherwise single int
                    if "," in knn_text:
                        knn_list = [int(k.strip()) for k in knn_text.split(",")]
                    else:
                        knn_list = [int(knn_text)]

                except ValueError as exc:
                    raise ValueError(
                        "Invalid KNN value. Please enter a positive integer."
                    ) from exc
                calculate_phenotype_similarity_knn = (
                    self.calculate_phenotype_similarity_knn_checkbox.isChecked()
                )

            if use_touching_neighbours_3d:
                try:
                    touching_dilation_um = float(self.touching_dilation_um_input.text())
                except ValueError as exc:
                    raise ValueError(
                        "Invalid 3D touching dilation value. Please enter a positive number in um."
                    ) from exc
                if touching_dilation_um <= 0:
                    raise ValueError(
                        "Invalid 3D touching dilation value. Please enter a positive number in um."
                    )
                calculate_phenotype_similarity_touching = (
                    self.calculate_phenotype_similarity_touching_checkbox.isChecked()
                )

        # Per axis: a filled box overrides that axis, a blank one keeps whatever
        # the file says. parse_override raises a message meant for the user.
        user_voxel_size = parse_override(
            self.voxel_size_z_input.text(),
            self.voxel_size_y_input.text(),
            self.voxel_size_x_input.text(),
        )
        return (
            extra_props,
            advanced_statistics_only,
            measure_regions,
            cytoplasm_size,
            save_measurement_mask,
            user_voxel_size,
            calculate_neighbour_statistics,
            use_knn_neighbours,
            knn_list,
            calculate_phenotype_similarity_knn,
            use_touching_neighbours_3d,
            touching_dilation_um,
            calculate_phenotype_similarity_touching,
        )

    def _create_cropping_layout(self):
        layout = QVBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignTop)
        self.crop_config_btn = QPushButton("Show cropping settings")
        self.crop_config_btn.setCheckable(True)
        self.crop_config_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        layout.addWidget(self.crop_config_btn)

        self.crop_config_widget = QWidget()
        self.crop_config_widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self.crop_config_widget.setLayout(self._create_cropping_settings_layout())
        self.crop_config_widget.setVisible(False)
        layout.addWidget(self.crop_config_widget)

        self.crop_config_btn.toggled.connect(self._toggle_crop_layout)
        self.crop_config_btn.setChecked(True)  # start expanded
        return layout

    def _toggle_crop_layout(self, checked):
        self.crop_config_widget.setVisible(checked)
        self.crop_config_btn.setText(
            "Hide cropping settings" if checked else "Show cropping settings"
        )

    def _create_cropping_settings_layout(self):
        layout = QVBoxLayout()
        layout.setSpacing(0)

        crop_row = QHBoxLayout()
        crop_row.addWidget(QLabel("Cropping before segmentation:"))
        crop_row.addWidget(self._info_label(
            "Timelapses are cropped automatically; fixed samples can use manual or automatic cropping.\n"
            "Samples that already have a cropped file reuse it, unless you choose to crop them again."
        ))
        crop_row.addStretch()
        layout.addLayout(crop_row)

        # One crop mode instead of separate crop/skip checkboxes. The first option's
        # label and the status line adapt to the selected samples (_update_crop_status).
        self.crop_new_radio = QRadioButton("Crop samples")
        self.crop_again_radio = QRadioButton(
            "Crop all selected samples again (replaces existing cropped files)"
        )
        self.no_crop_radio = QRadioButton("Don't crop")
        self.crop_mode_group = QButtonGroup(self)
        for button in (self.crop_new_radio, self.crop_again_radio, self.no_crop_radio):
            self.crop_mode_group.addButton(button)
            layout.addWidget(button)
        self.crop_new_radio.setChecked(True)
        self.crop_mode_group.buttonToggled.connect(self._on_crop_mode_changed)

        self.crop_status_label = QLabel("")
        self.crop_status_label.setWordWrap(True)
        self.crop_status_label.setContentsMargins(20, 2, 0, 2)
        self.crop_status_label.setStyleSheet("color: gray;")
        layout.addWidget(self.crop_status_label)

        # All suboptions in one indented container — shown/hidden together
        self.crop_suboptions_widget = QWidget()
        sub_layout = QVBoxLayout(self.crop_suboptions_widget)
        sub_layout.setContentsMargins(20, 2, 0, 0)
        sub_layout.setSpacing(4)

        self.manual_crop_fixed_checkbox = QCheckBox(
            "For fixed samples: use manual cropping"
        )
        self.manual_crop_fixed_checkbox.setChecked(False)
        sub_layout.addWidget(self.manual_crop_fixed_checkbox)

        combo_row = QHBoxLayout()
        combo_row.setContentsMargins(0, 2, 0, 0)
        combo_row.addWidget(QLabel("Save cropped files as:"))
        self.save_crop_as = QComboBox()
        self.save_crop_as.addItems([".tif", ".ims"])
        combo_row.addWidget(self.save_crop_as)
        combo_row.addWidget(self._info_label(
            "OME-TIFF (.tif) is the default: it carries voxel size and timing, "
            "opens anywhere, and has no size limit.\n"
            "Choose .ims only if you work in Imaris."
        ))
        combo_row.addStretch()
        sub_layout.addLayout(combo_row)

        layout.addWidget(self.crop_suboptions_widget)
        self._on_crop_mode_changed()

        return layout

    def _crop_mode(self):
        if self.no_crop_radio.isChecked():
            return "none"
        if self.crop_again_radio.isChecked():
            return "recrop"
        return "crop"

    def _on_crop_mode_changed(self, *_):
        self.crop_suboptions_widget.setVisible(self._crop_mode() != "none")
        self._update_crop_status()

    def _update_crop_status(self):
        """Adapt the crop options and status line to the selected samples' cropped files."""
        if not hasattr(self, "crop_status_label"):
            return
        paths = self.get_sample_path_list() if self.sample_list is not None else []
        total = len(paths)
        cropped = sum(1 for path in paths if find_cropped_files(path))
        mode = self._crop_mode()

        if cropped == 0:
            self.crop_new_radio.setText("Crop samples")
        elif cropped == total:
            self.crop_new_radio.setText("Use existing cropped files")
        else:
            self.crop_new_radio.setText("Crop samples, reusing existing cropped files")

        # Cropping "again" only means something when there is a cropped file to replace.
        # Without one both options do the same, so fall back rather than hide a checked one.
        self.crop_again_radio.setVisible(cropped > 0)
        if cropped == 0 and mode == "recrop":
            self.crop_new_radio.setChecked(True)  # re-enters via _on_crop_mode_changed
            return

        samples = "the selected sample" if total == 1 else f"all {total} selected samples"
        if total == 0 or mode == "none":
            text = ""
        elif mode == "recrop":
            text = f"{samples.capitalize()} will be cropped again, replacing existing cropped files."
        elif cropped == 0:
            text = f"No cropped files found yet, so {samples} will be cropped."
        elif cropped == total:
            verb = "has" if total == 1 else "have"
            text = f"{samples.capitalize()} already {verb} a cropped file, which will be reused."
        else:
            text = (
                f"{cropped} of {total} selected samples already have a cropped file, which will "
                f"be reused; the other {total - cropped} will be cropped."
            )
        self.crop_status_label.setText(text)
        self.crop_status_label.setVisible(bool(text))

        # When every sample reuses its cropped file, nothing gets cropped.
        self.crop_suboptions_widget.setEnabled(not (mode == "crop" and total and cropped == total))

    def _create_phenotype_settings_layout(self):
        layout = QVBoxLayout()
        layout.setSpacing(0)
        self.phenotype_settings_btn = QPushButton(
            "Show phenotype/cell-type calling settings"
        )
        self.phenotype_settings_btn.setCheckable(True)
        self.phenotype_settings_btn.toggled.connect(self._toggle_phenotype_settings)
        self.phenotype_settings_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        layout.addWidget(self.phenotype_settings_btn)

        # Create phenotype settings widget
        self.phenotype_settings_widget = QWidget()
        self.phenotype_settings_widget.setLayout(self._create_phenotype_settings())
        self.phenotype_settings_widget.setVisible(False)
        layout.addWidget(self.phenotype_settings_widget)
        return layout

    def _toggle_phenotype_settings(self, checked):
        self.phenotype_settings_widget.setVisible(checked)
        self.phenotype_settings_btn.setText(
            "Hide phenotype/cell-type calling settings" if checked else "Show phenotype/cell-type calling settings"
        )

    def _create_phenotype_settings(self):
        """Add UI elements for phenotype/cell-type calling settings here"""
        layout = QVBoxLayout()
        layout.setSpacing(10)

        self.do_phenotype_calling_checkbox = QCheckBox(
            "Perform phenotype/cell-type calling"
        )
        self.do_phenotype_calling_checkbox.toggled.connect(self._toggle_phenotype_daughters)
        layout.addWidget(self.do_phenotype_calling_checkbox)

        # All daughter settings — indented, shown only when parent checkbox is checked
        self.phenotype_daughters_widget = QWidget()
        daughters_layout = QVBoxLayout(self.phenotype_daughters_widget)
        daughters_layout.setContentsMargins(20, 4, 0, 0)
        daughters_layout.setSpacing(8)

        self.create_split_phenotype_mask = QCheckBox(
            "Create an additional segmentation mask file that splits the called phenotypes/cell-types into different channels"
        )
        daughters_layout.addWidget(self.create_split_phenotype_mask)

        layout2 = QHBoxLayout()
        layout2.addWidget(QLabel("Calculating phenotype/cell-type based on"))
        self.phenotype_1 = QComboBox()
        self.phenotype_2 = QComboBox()
        self._update_phenotype_combos()
        # Each channel is measured in its own region, independent of the regions
        # ticked in the advanced statistics settings.
        self.phenotype_1_region = self._region_combo()
        self.phenotype_2_region = self._region_combo()
        layout2.addWidget(self.phenotype_1)
        layout2.addWidget(QLabel("in"))
        layout2.addWidget(self.phenotype_1_region)
        layout2.addWidget(QLabel("VS"))
        layout2.addWidget(self.phenotype_2)
        layout2.addWidget(QLabel("in"))
        layout2.addWidget(self.phenotype_2_region)
        layout2.addStretch()
        daughters_layout.addLayout(layout2)

        layout3 = QHBoxLayout()
        layout3.addWidget(QLabel("Cutoff calculation method:"))
        self.calculate_cutoff = QComboBox()
        self.calculate_cutoff.addItems(
            [
                "Automatic",
                "Fixed at 0",
                "Custom cutoff value",
            ]
        )
        self.calculate_cutoff.currentTextChanged.connect(self._toggle_custom_cutoff)
        layout3.addWidget(self.calculate_cutoff)
        layout3.addWidget(self._info_label(
            "Automatic: finds the cutoff that best separates two populations — best when both are roughly equal in size.\n"
            "Fixed at 0: uses a log10 ratio of 0 (equal intensity) as the cutoff — best when one population is clearly positive and the other negative.\n"
            "Custom: lets you set the cutoff manually as a log10 ratio value."
        ))
        layout3.addStretch()
        daughters_layout.addLayout(layout3)

        custom_cutoff_container = QWidget()
        layout_custom_cutoff = QHBoxLayout(custom_cutoff_container)
        layout_custom_cutoff.setContentsMargins(20, 0, 0, 0)
        self.custom_cutoff_label = QLabel("Custom cutoff value (log10 ratio):")
        self.custom_cutoff_label.setVisible(False)
        self.custom_cutoff_input = QLineEdit("0.0")
        self.custom_cutoff_input.setPlaceholderText("e.g. -0.25")
        self.custom_cutoff_input.setVisible(False)
        layout_custom_cutoff.addWidget(self.custom_cutoff_label)
        layout_custom_cutoff.addWidget(self.custom_cutoff_input)
        layout_custom_cutoff.addStretch()
        daughters_layout.addWidget(custom_cutoff_container)

        layout4 = QHBoxLayout()
        layout4.addWidget(QLabel("Base phenotype/cell-type on:"))
        self.raw_or_background_subtracted = QComboBox()
        self.raw_or_background_subtracted.addItems(
            ["Background subtracted mean intensity values", "Raw mean intensity values"]
        )
        layout4.addWidget(self.raw_or_background_subtracted)
        layout4.addWidget(self._info_label(
            "Raw: uses the raw pixel intensity values measured inside each nucleus/cell.\n"
            "Background subtracted: subtracts an estimated background (the median pixel value of the whole image) from the raw value, "
            "reducing the influence of uneven illumination or autofluorescence."
        ))
        layout4.addStretch()
        daughters_layout.addLayout(layout4)

        self.phenotype_daughters_widget.setVisible(False)
        layout.addWidget(self.phenotype_daughters_widget)

        return layout

    def _toggle_phenotype_daughters(self, checked):
        self.phenotype_daughters_widget.setVisible(checked)

    def _toggle_custom_cutoff(self, selected_cutoff_method):
        show_custom = "custom cutoff" in selected_cutoff_method.lower()
        self.custom_cutoff_input.setVisible(show_custom)
        self.custom_cutoff_label.setVisible(show_custom)

    def _create_run_segmentation_layout(self):
        layout = QVBoxLayout()
        layout.setSpacing(5)
        self.segment_tab_label = QLabel("Selected 0 samples for segmentation")
        layout.addWidget(self.segment_tab_label)
        self.run_segmentation_btn = QPushButton("Run segmentation")
        self.run_segmentation_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.run_segmentation_btn.clicked.connect(lambda: self.start_segmentation())
        self.stop_segmentation_btn = QPushButton("Stop after current sample")
        self.stop_segmentation_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.stop_segmentation_btn.setVisible(False)
        self.stop_segmentation_btn.clicked.connect(self._request_stop)
        self.continue_segmentation_btn = QPushButton("Continue segmentation")
        self.continue_segmentation_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.continue_segmentation_btn.setVisible(False)
        self.continue_segmentation_btn.clicked.connect(self._continue_segmentation)
        save_config_btn = QPushButton("Save configuration")
        save_config_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        save_config_btn.clicked.connect(self.save_configuration)
        load_config_btn = QPushButton("Load configuration")
        load_config_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        load_config_btn.clicked.connect(self.load_configuration)
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_row.addWidget(self.run_segmentation_btn)
        btn_row.addWidget(self.continue_segmentation_btn)
        btn_row.addWidget(self.stop_segmentation_btn)
        btn_row.addStretch()
        btn_row.addWidget(self._info_label(
            "Save all current settings to a JSON file so you can reload them later.\n"
            "Useful for repeating the same analysis across experiments without re-entering every setting."
        ))
        btn_row.addWidget(save_config_btn)
        btn_row.addWidget(load_config_btn)
        layout.addLayout(btn_row)
        return layout

    def _update_segment_label(self):
        """Update the segment tab label with selected samples count"""
        count = len(self.get_selected_samples())
        self.segment_tab_label.setText(f"Selected {count} samples for segmentation")
        self._update_crop_status()

    def _create_progress_bar(self):
        layout = QVBoxLayout()
        layout.setSpacing(0)
        self.progressbar = QProgressBar()
        self.progressbar.setVisible(
            False
        )  # Initially hidden, show when running segmentation
        layout.addWidget(self.progressbar)
        self.segmentation_summary_label = QLabel("")
        self.segmentation_summary_label.setVisible(
            False
        )  # Initially hidden, show after segmentation
        layout.addWidget(self.segmentation_summary_label)
        # Separate line for skipped samples: the summary label above is
        # rewritten every second by the watchdog, so failures need their own.
        self.segmentation_failures_label = QLabel("")
        self.segmentation_failures_label.setStyleSheet("color: #FFA500;")
        self.segmentation_failures_label.setWordWrap(True)
        self.segmentation_failures_label.setVisible(False)
        layout.addWidget(self.segmentation_failures_label)
        return layout

    def _create_advanced_settings_layout(self):
        layout = QVBoxLayout()
        layout.setSpacing(0)
        self.adv_settings_btn = QPushButton("Show advanced segmentation settings")
        self.adv_settings_btn.setCheckable(True)
        self.adv_settings_btn.toggled.connect(self._toggle_advanced_settings)
        self.adv_settings_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        layout.addWidget(self.adv_settings_btn)

        # Create advanced settings widget
        self.advanced_settings_widget = QWidget()
        self.advanced_settings_widget.setLayout(self._create_advanced_settings())
        self.advanced_settings_widget.setVisible(False)
        layout.addWidget(self.advanced_settings_widget)
        return layout

    def _toggle_advanced_settings(self, checked):
        self.advanced_settings_widget.setVisible(checked)
        self.adv_settings_btn.setText(
            "Hide advanced segmentation settings" if checked else "Show advanced segmentation settings"
        )

    def _create_advanced_settings(self):
        """Add UI elements for advanced settings here"""
        self.advanced_layout = QVBoxLayout()
        self.advanced_layout.setSpacing(10)
        self.advanced_layout.addLayout(self._create_model_settings())
        
        breaking_threshold_layout = QHBoxLayout()
        breaking_threshold_layout.addWidget(
            QLabel("Breaking threshold for cell stitching:")
        )
        self.breaking_threshold_input = QLineEdit("3")
        self.breaking_threshold_input.setMaximumWidth(90)
        breaking_threshold_layout.addWidget(self.breaking_threshold_input)
        breaking_threshold_layout.addWidget(self._info_label(
            "Controls how aggressively the 3D stitching splits nuclei detected across Z-slices.\n"
            "Lower value → more nuclei found, each smaller (less merging).\n"
            "Higher value → fewer, larger nuclei (more merging across slices).\n"
            "Default (2.5) works well for most datasets."
        ))
        breaking_threshold_layout.addStretch()
        self.advanced_layout.addLayout(breaking_threshold_layout)

        size_filter_layout = QHBoxLayout()
        size_filter_layout.addWidget(QLabel("2D size filter multiplier:"))
        self.size_2d_filter_input = QLineEdit("15")
        self.size_2d_filter_input.setMaximumWidth(90)
        size_filter_layout.addWidget(self.size_2d_filter_input)
        size_filter_layout.addWidget(self._info_label(
            "Filters out 2D segmented cells larger than this multiple of the median 2D cell size.\n"
            "CellposeSAM can produce very large detections in empty slices because its size agnostic — this removes them before 3D stitching.\n"
            "Higher value → less filtering (keeps more large cells).\n"
            "Lower value → more filtering (removes large cells sooner).\n"
            "Default (15) works well for most datasets."
        ))
        size_filter_layout.addStretch()
        self.advanced_layout.addLayout(size_filter_layout)

        voxel_label_row = QHBoxLayout()
        voxel_label_row.addWidget(QLabel("Voxel size override (Z / X / Y in µm):"))
        voxel_label_row.addWidget(self._info_label(
            "Only needed if your image file has incorrect or missing spatial metadata.\n"
            "Each axis is separate: fill in the ones you want to set, and leave "
            "the rest blank to read them from the file."
        ))
        voxel_label_row.addStretch()
        self.advanced_layout.addLayout(voxel_label_row)
        layout_voxel = QHBoxLayout()
        layout_voxel.addWidget(QLabel("Z:"))
        self.voxel_size_z_input = QLineEdit()
        self.voxel_size_z_input.setPlaceholderText("auto")
        self.voxel_size_z_input.setMaximumWidth(90)
        layout_voxel.addWidget(self.voxel_size_z_input)
        layout_voxel.addWidget(QLabel("X:"))
        self.voxel_size_x_input = QLineEdit()
        self.voxel_size_x_input.setPlaceholderText("auto")
        self.voxel_size_x_input.setMaximumWidth(90)
        layout_voxel.addWidget(self.voxel_size_x_input)
        layout_voxel.addWidget(QLabel("Y:"))
        self.voxel_size_y_input = QLineEdit()
        self.voxel_size_y_input.setPlaceholderText("auto")
        self.voxel_size_y_input.setMaximumWidth(90)
        layout_voxel.addWidget(self.voxel_size_y_input)
        layout_voxel.addStretch()
        self.advanced_layout.addLayout(layout_voxel)

        self.advanced_layout.addLayout(self._create_save_individual_files())

        return self.advanced_layout

    def _create_channel_config_layout(self):
        layout = QVBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignTop)
        self.channel_config_btn = QPushButton("Show channel configuration")
        self.channel_config_btn.setCheckable(True)
        self.channel_config_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        btn_row.addWidget(self.channel_config_btn)
        btn_row.addWidget(self._info_label(
            "Configure ALL channels present in your image/movie — not only the channel(s) you want to segment."
        ))
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.channel_config_widget = QWidget()
        self.channel_config_widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self.channel_config_widget.setLayout(self._create_channel_settings_menu())
        self.channel_config_widget.setVisible(False)
        layout.addWidget(self.channel_config_widget)

        self.channel_config_btn.toggled.connect(self._toggle_channel_config)
        self.channel_config_btn.setChecked(True)  # start expanded
        return layout

    def _toggle_channel_config(self, checked):
        self.channel_config_widget.setVisible(checked)
        self.channel_config_btn.setText(
            "Hide channel configuration" if checked else "Show channel configuration"
        )

    def _create_channel_settings_menu(self):
        """Add UI elements for channel settings here"""
        layout = QVBoxLayout()
        layout.setSpacing(10)

        # Use grid layout for aligned columns
        self.channel_grid = QGridLayout()
        self.channel_grid.setSpacing(10)

        # Header row
        for text, col in [("Channel", 0), ("Type", 1)]:
            lbl = QLabel(text)
            lbl.setStyleSheet("font-weight: bold;")
            self.channel_grid.addWidget(lbl, 0, col)
        name_header_widget = QWidget()
        name_header_row = QHBoxLayout(name_header_widget)
        name_header_row.setContentsMargins(0, 0, 0, 0)
        name_header_row.setSpacing(2)
        name_lbl = QLabel("Name")
        name_lbl.setStyleSheet("font-weight: bold;")
        name_header_row.addWidget(name_lbl)
        name_header_row.addWidget(
            self._info_label("Name is only used to generate accurate file and column names in the output data.")
        )
        name_header_row.addStretch()
        self.channel_grid.addWidget(name_header_widget, 0, 2)

        # Initial channel line
        self.channel_count = 1
        self.channel_widgets = {}  # Store widgets for deletion
        self._add_channel_grid_row(self.channel_count)

        layout.addLayout(self.channel_grid)

        # Add/Delete channel buttons
        button_layout = QHBoxLayout()
        add_channel_btn = QPushButton("Add channel")
        add_channel_btn.clicked.connect(self.add_channel)
        delete_channel_btn = QPushButton("Delete channel")
        delete_channel_btn.clicked.connect(self.delete_channel)
        button_layout.addWidget(add_channel_btn)
        button_layout.addWidget(delete_channel_btn)
        layout.addLayout(button_layout)

        return layout

    def _add_channel_grid_row(self, channel_number):
        """Add a new row to the channel grid"""
        row = self.channel_grid.rowCount()

        # Channel number label
        label = QLabel(f"{channel_number}:")
        self.channel_grid.addWidget(label, row, 0)

        # Type combobox
        combo_box = QComboBox()
        combo_box.addItems(["Nuclei marker", "Other channel"])
        self.channel_grid.addWidget(combo_box, row, 1)

        # Channel name input
        line_edit = QLineEdit(f"Channel_{channel_number}")
        line_edit.editingFinished.connect(self._update_phenotype_combos)
        self.channel_grid.addWidget(line_edit, row, 2)

        # Store widgets for later deletion
        self.channel_widgets[channel_number] = (label, combo_box, line_edit)

    def _update_phenotype_combos(self):
        """Refresh phenotype comboboxes to match current channel names."""
        if not hasattr(self, "phenotype_1"):
            return
        channel_names = [
            line_edit.text()
            for _, (_, _, line_edit) in sorted(self.channel_widgets.items())
        ]
        for combo in (self.phenotype_1, self.phenotype_2):
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(channel_names)
            idx = combo.findText(current)
            combo.setCurrentIndex(idx if idx >= 0 else 0)
            combo.blockSignals(False)

    def add_channel(self):
        """Add a new channel settings line"""
        self.channel_count += 1
        self._add_channel_grid_row(self.channel_count)
        self._update_phenotype_combos()

    def delete_channel(self):
        """Delete the last channel settings line"""
        if self.channel_count > 1:
            # Delete widgets from grid
            label, combo_box, line_edit = self.channel_widgets[self.channel_count]
            label.deleteLater()
            combo_box.deleteLater()
            line_edit.deleteLater()

            # Remove from storage
            del self.channel_widgets[self.channel_count]
            self.channel_count -= 1
            self._update_phenotype_combos()

    def _create_model_settings(self):
        model_setting_layout = QHBoxLayout()
        model_setting_layout.setSpacing(6)
        model_setting_layout.addWidget(QLabel("2D Segmentation model:"))
        self.model_combo_box = QComboBox()
        self._refresh_model_options()
        model_setting_layout.addWidget(self.model_combo_box)
        upload_button = QPushButton("Upload custom model")
        upload_button.clicked.connect(self._upload_custom_model)
        model_setting_layout.addWidget(upload_button)
        extra_models_button = QPushButton("Download extra pretrained models")
        extra_models_button.clicked.connect(self._download_extra_models)
        model_setting_layout.addWidget(extra_models_button)
        model_setting_layout.addWidget(self._info_label(
            "Pretrained 2D Cellpose segmentation models used to detect nuclei in each Z-slice.\n"
            "The slices are then stitched into a full 3D segmentation.\n"
            "You can upload a custom Cellpose model trained on your own data,\n"
            f"or use the built-in '{CELLPOSE_SAM_MODEL_NAME}' (always the last option)."
        ))
        model_setting_layout.addStretch()
        return model_setting_layout

    def _get_models_dir(self):
        return os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

    def _is_sam_support_file(self, entry_name):
        lower_name = entry_name.lower()
        return lower_name in {"sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"}

    def _refresh_model_options(self, select_name=None):
        models_dir = self._get_models_dir()
        os.makedirs(models_dir, exist_ok=True)

        if select_name is None and hasattr(self, "model_combo_box"):
            select_name = self.model_combo_box.currentText()

        model_entries = []
        for entry in sorted(os.listdir(models_dir)):
            # .part files are downloads still in flight, not usable models.
            if self._is_sam_support_file(entry) or entry.endswith(".part"):
                continue
            full_path = os.path.join(models_dir, entry)
            if os.path.isdir(full_path) or os.path.isfile(full_path):
                model_entries.append((entry, full_path))

        self.model_combo_box.blockSignals(True)
        self.model_combo_box.clear()
        for display_name, full_path in model_entries:
            self.model_combo_box.addItem(display_name, full_path)

        # The Cellpose-SAM foundation model always ships with the environment,
        # so it is always offered as the last option.
        self.model_combo_box.addItem(CELLPOSE_SAM_MODEL_NAME, CELLPOSE_SAM_MODEL_NAME)

        if select_name:
            idx = self.model_combo_box.findText(select_name)
            self.model_combo_box.setCurrentIndex(idx if idx >= 0 else 0)
        else:
            self.model_combo_box.setCurrentIndex(0)
        self.model_combo_box.blockSignals(False)

    def _download_extra_models(self):
        # Models published alongside the required ones but not fetched by default,
        # so a normal install does not pay for weights it will never use.
        try:
            extras = optional_assets()
        except Exception as e:
            QMessageBox.warning(
                self, "Extra models", f"Could not reach the model release:\n{e}"
            )
            return

        if not extras:
            QMessageBox.information(
                self, "Extra models", "No extra pretrained models are available."
            )
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Download extra pretrained models")
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel(
            "Pretrained models for other sample types. Selected models are saved\n"
            "to the models folder and then appear in the 2D segmentation model list."
        ))

        boxes = []
        for asset in extras:
            label = f"{asset['name']}  ({asset['size'] / 1048576:.0f} MB)"
            box = QCheckBox(
                f"{label} — already downloaded" if asset["installed"] else label
            )
            box.setEnabled(not asset["installed"])
            layout.addWidget(box)
            boxes.append((box, asset))

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        selected = [asset for box, asset in boxes if box.isChecked()]
        if not selected:
            return

        self._extra_downloader = ModelDownloader(assets=selected)
        self._extra_downloader.progress.connect(self._on_extra_progress)
        self._extra_downloader.completed.connect(self._on_extra_done)
        self._extra_downloader.error.connect(self._on_extra_error)
        self._extra_downloader.start()

    def _on_extra_progress(self, msg):
        self._install_btn.setVisible(False)
        self._update_label.setText(msg)
        self._update_banner.setVisible(True)

    def _on_extra_done(self):
        self._install_btn.setVisible(True)
        self._update_banner.setVisible(False)
        self._refresh_model_options()

    def _on_extra_error(self, msg):
        print(f"[models] extra download failed: {msg}")
        self._update_label.setText(f"Could not download extra models: {msg}")
        self._update_banner.setVisible(True)

    def _upload_custom_model(self):
        file_filter = "Model Files (*.pt *.pth *.onnx *.npy);;All Files (*)"
        src_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select custom model file",
            "",
            file_filter,
            "All Files (*)",
        )
        if not src_path:
            return

        models_dir = self._get_models_dir()
        os.makedirs(models_dir, exist_ok=True)

        src_name = os.path.basename(src_path)
        dst_path = os.path.join(models_dir, src_name)

        if os.path.abspath(src_path) != os.path.abspath(dst_path):
            if os.path.exists(dst_path):
                stem, ext = os.path.splitext(src_name)
                suffix = 1
                while os.path.exists(os.path.join(models_dir, f"{stem}_{suffix}{ext}")):
                    suffix += 1
                dst_path = os.path.join(models_dir, f"{stem}_{suffix}{ext}")
            shutil.copy2(src_path, dst_path)

        self._refresh_model_options(select_name=os.path.basename(dst_path))

    def _create_save_individual_files(self):
        layout = QVBoxLayout()
        self.save_frames_checkbox = QCheckBox(
            "Save individual frames per timepoint (optional)"
        )
        self.save_frames_checkbox.setChecked(False)
        layout.addWidget(self.save_frames_checkbox)
        self.save_segmentation_checkbox = QCheckBox(
            "Save individual segmentation masks per timepoint (optional)"
        )
        self.save_segmentation_checkbox.setChecked(False)
        layout.addWidget(self.save_segmentation_checkbox)
        return layout

    def get_channel_settings(self):
        """Extract channel names and types from the grid"""
        channel_names = []
        channel_types = []
        for channel_num in sorted(self.channel_widgets.keys()):
            _, combo_box, line_edit = self.channel_widgets[channel_num]
            channel_types.append(combo_box.currentText())
            channel_names.append(line_edit.text())
        return channel_names, channel_types

    def get_model_path(self):
        """Get the model path based on selected model.

        Returns the sentinel CELLPOSE_SAM_MODEL_NAME when the built-in
        Cellpose-SAM foundation model is selected (it has no file on disk).
        """
        if self.model_combo_box.currentText() == CELLPOSE_SAM_MODEL_NAME:
            return CELLPOSE_SAM_MODEL_NAME

        model_path = self.model_combo_box.currentData()
        if model_path and os.path.exists(model_path):
            return model_path

        selected_name = self.model_combo_box.currentText()
        if not selected_name:
            return None

        fallback_path = os.path.join(self._get_models_dir(), selected_name)
        return fallback_path if os.path.exists(fallback_path) else None

    def get_sample_path_list(self):
        """Get list of selected samples"""
        samples = self.get_selected_samples()
        base_path = self.path_text.text()
        return [os.path.join(base_path, sample) for sample in samples]

    def get_breaking_threshold(self):
        try:
            return float(self.breaking_threshold_input.text())
        except ValueError:
            return 2.5

    def get_size_2d_filter_multiplier(self):
        try:
            return float(self.size_2d_filter_input.text())
        except ValueError:
            return 15

    def get_phenotype_calling_settings(self):
        """Get phenotype calling settings"""
        do_calling = self.do_phenotype_calling_checkbox.isChecked()
        channel_1 = self.phenotype_1.currentText()
        channel_2 = self.phenotype_2.currentText()
        cutoff_method = self.calculate_cutoff.currentText()
        custom_cutoff = None
        if "custom cutoff" in cutoff_method.lower():
            try:
                custom_cutoff = float(self.custom_cutoff_input.text())
            except ValueError as exc:
                raise ValueError(
                    "Invalid custom cutoff value. Please enter a valid float."
                ) from exc
        raw_or_background_subtracted = self.raw_or_background_subtracted.currentText()
        run_mode = self.run_mode_combo.currentText()
        phenotype_calling_only = run_mode in ("Phenotype/cell-type calling only", "Statistics + phenotype/cell-type calling")
        create_split_phenotype_mask = self.create_split_phenotype_mask.isChecked()
        return (
            do_calling,
            channel_1,
            channel_2,
            cutoff_method,
            custom_cutoff,
            raw_or_background_subtracted,
            phenotype_calling_only,
            create_split_phenotype_mask,
            self.phenotype_1_region.currentData(),
            self.phenotype_2_region.currentData(),
        )

    def start_segmentation(self, sample_override=None, index_offset=0, total_count=None):
        """Start segmentation in a separate thread"""
        sample_path_list = sample_override if sample_override is not None else self.get_sample_path_list()
        if total_count is None:
            total_count = len(sample_path_list)
        if not sample_path_list:
            self.segmentation_summary_label.setText(
                "No samples selected for segmentation."
            )
            self.segmentation_summary_label.setVisible(True)
            return

        self.progressbar.setRange(0, 100)
        self.progressbar.setVisible(True)
        self.progressbar.setValue(0)
        self.run_segmentation_btn.setEnabled(False)
        self.continue_segmentation_btn.setVisible(False)
        self.segmentation_summary_label.setVisible(False)
        if sample_override is None:
            # Fresh run: clear failures. A continuation keeps the earlier ones.
            self._failed_samples = []
            self.segmentation_failures_label.setVisible(False)
            self.segmentation_failures_label.setText("")
        QApplication.processEvents()

        model_path = self.get_model_path()
        if model_path is None:
            self.segmentation_error(
                "No segmentation model found. Please upload a model in the advanced segmentation settings."
            )
            return
        channel_names, channel_types = self.get_channel_settings()
        breaking_threshold = self.get_breaking_threshold()
        size_2d_filter_multiplier = self.get_size_2d_filter_multiplier()
        try:
            (
                do_phenotype_calling,
                phenotype_1,
                phenotype_2,
                cutoff_method,
                custom_cutoff,
                raw_or_background_subtracted,
                phenotype_calling_only,
                create_split_phenotype_mask,
                phenotype_1_region,
                phenotype_2_region,
            ) = self.get_phenotype_calling_settings()
        except ValueError as e:
            self.segmentation_error(str(e))
            return
        crop_mode = self._crop_mode()
        do_crop_sample = crop_mode != "none"
        manual_crop_fixed = self.manual_crop_fixed_checkbox.isChecked()
        save_crop_as = self.save_crop_as.currentText()
        save_frames = self.save_frames_checkbox.isChecked()
        save_segmentation = self.save_segmentation_checkbox.isChecked()
        try:
            (
                extra_props,
                advanced_statistics_only,
                measure_regions,
                cytoplasm_size,
                save_measurement_mask,
                user_voxel_size,
                calculate_neighbour_statistics,
                use_knn_neighbours,
                knn_list,
                calculate_phenotype_similarity_knn,
                use_touching_neighbours_3d,
                touching_dilation_um,
                calculate_phenotype_similarity_touching,
            ) = self.get_advanced_statistics_settings()
        except ValueError as e:
            self.segmentation_error(str(e))
            return

        manually_cropped_fixed_samples = []
        if (
            do_crop_sample
            and manual_crop_fixed
            and not phenotype_calling_only
            and not advanced_statistics_only
        ):
            try:
                from main_functions.crop_sample import crop_sample

                # Manual fixed-sample cropping happens in UI thread to allow interaction.
                self.segmentation_summary_label.setStyleSheet("")
                self.segmentation_summary_label.setText(
                    "Manual cropping in progress — please interact with the crop window..."
                )
                self.segmentation_summary_label.setVisible(True)
                self.progressbar.setRange(0, 0)
                QApplication.processEvents()

                # Only samples that will actually be cropped; the rest reuse their file.
                for sample_path in sample_path_list:
                    if not needs_crop(sample_path, crop_mode):
                        continue
                    was_cropped = crop_sample(
                        sample_path,
                        channel_types,
                        organoid_model=None,
                        save_as=save_crop_as,
                        manual_fixed=True,
                        fixed_only=True,
                        user_voxel_size=user_voxel_size,
                    )
                    if was_cropped:
                        remove_stale_cropped_files(sample_path, save_crop_as)
                        manually_cropped_fixed_samples.append(sample_path)

                # Restore determinate progress for worker phase.
                self.progressbar.setRange(0, 100)
                self.progressbar.setValue(0)
                QApplication.processEvents()
            except Exception as e:
                self.segmentation_error(str(e))
                return

        self.worker = SegmentationWorker(
            sample_path_list,
            model_path,
            channel_names,
            channel_types,
            breaking_threshold,
            size_2d_filter_multiplier,
            do_phenotype_calling,
            phenotype_calling_only,
            create_split_phenotype_mask,
            phenotype_1,
            phenotype_2,
            cutoff_method,
            custom_cutoff,
            raw_or_background_subtracted,
            do_crop_sample,
            manual_crop_fixed,
            crop_mode,
            save_crop_as,
            save_frames,
            save_segmentation,
            extra_props,
            advanced_statistics_only,
            measure_regions,
            cytoplasm_size,
            save_measurement_mask,
            user_voxel_size,
            calculate_neighbour_statistics,
            use_knn_neighbours,
            knn_list,
            calculate_phenotype_similarity_knn,
            use_touching_neighbours_3d,
            touching_dilation_um,
            calculate_phenotype_similarity_touching,
            manually_cropped_fixed_samples,
            index_offset=index_offset,
            total_count=total_count,
            phenotype_1_region=phenotype_1_region,
            phenotype_2_region=phenotype_2_region,
        )
        self.worker.progress_updated.connect(self.progressbar.setValue)
        self.worker.finished.connect(self.segmentation_finished)
        self.worker.stopped.connect(self.segmentation_stopped)
        self.worker.error_occurred.connect(self.segmentation_error)
        self.worker.heartbeat.connect(self._on_worker_heartbeat)
        self.worker.sample_failed.connect(self._on_sample_failed)
        self.stop_segmentation_btn.setVisible(True)
        self.stop_segmentation_btn.setEnabled(True)
        self.worker.start()
        import time as _time
        self._segmentation_running = True
        self._last_heartbeat_msg = "Starting..."
        self._last_heartbeat_time = _time.time()
        self._watchdog_timer = QTimer()
        self._watchdog_timer.timeout.connect(self._watchdog_tick)
        self._watchdog_timer.start(1_000)

    def segmentation_finished(self, elapsed_time):
        """Called when segmentation finishes"""
        self._segmentation_running = False
        if hasattr(self, "_watchdog_timer"):
            self._watchdog_timer.stop()
        self.run_segmentation_btn.setEnabled(True)
        self.stop_segmentation_btn.setVisible(False)
        self.stop_segmentation_btn.setEnabled(True)
        self.stop_segmentation_btn.setText("Stop after current sample")
        self.continue_segmentation_btn.setVisible(False)
        self.progressbar.setRange(0, 100)
        self.progressbar.setValue(100)

        finish_time = datetime.now().strftime("%H:%M:%S")
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)

        failed_count = len(getattr(self, "_failed_samples", []))
        total = len(self.get_selected_samples())
        if failed_count:
            self.segmentation_summary_label.setStyleSheet("color: #FFA500;")
            summary_text = (
                f"Segmented {total - failed_count} of {total} samples "
                f"({failed_count} skipped after errors).\n"
                f"Segmentation completed at {finish_time} (took {minutes}m {seconds}s)"
            )
        else:
            self.segmentation_summary_label.setStyleSheet("color: #44BB44;")
            summary_text = f"Segmented {total} samples.\nSegmentation completed at {finish_time} (took {minutes}m {seconds}s)"
        self.segmentation_summary_label.setText(summary_text)
        self.segmentation_summary_label.setVisible(True)
        self._refresh_failures_label()
        self._update_crop_status()  # the run may have written new cropped files

    def segmentation_stopped(self, elapsed_time, stopped_idx):
        """Called when the user stops segmentation between samples"""
        self._segmentation_running = False
        self._update_crop_status()
        if hasattr(self, "_watchdog_timer"):
            self._watchdog_timer.stop()
        self.run_segmentation_btn.setEnabled(True)
        self.stop_segmentation_btn.setVisible(False)
        self.stop_segmentation_btn.setEnabled(True)
        self.stop_segmentation_btn.setText("Stop after current sample")
        self.progressbar.setRange(0, 100)
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        remaining = []
        continuation_offset = stopped_idx
        continuation_total = stopped_idx
        if hasattr(self, "worker"):
            remaining = self.worker.sample_path_list[stopped_idx:]
            continuation_offset = self.worker.index_offset + stopped_idx
            continuation_total = self.worker.total_count
        self._continuation_sample_paths = remaining
        self._continuation_index_offset = continuation_offset
        self._continuation_total_count = continuation_total
        remaining_count = len(remaining)
        self.continue_segmentation_btn.setVisible(remaining_count > 0)
        self.segmentation_summary_label.setStyleSheet("color: #FFA500;")
        remaining_text = f" {remaining_count} sample(s) remaining." if remaining_count > 0 else ""
        self.segmentation_summary_label.setText(
            f"Stopped by user after {minutes}m {seconds}s.{remaining_text}"
        )
        self.segmentation_summary_label.setVisible(True)
        self._refresh_failures_label()

    def segmentation_error(self, error_message):
        """Called when an error occurs during segmentation"""
        self._segmentation_running = False
        if hasattr(self, "_watchdog_timer"):
            self._watchdog_timer.stop()
        self.run_segmentation_btn.setEnabled(True)
        self.stop_segmentation_btn.setVisible(False)
        self.stop_segmentation_btn.setEnabled(True)
        self.stop_segmentation_btn.setText("Stop after current sample")
        self.continue_segmentation_btn.setVisible(False)
        self.progressbar.setVisible(False)

        self.segmentation_summary_label.setStyleSheet("color: #FF4444;")
        self.segmentation_summary_label.setText(f"Error: {error_message}")
        self.segmentation_summary_label.setVisible(True)
        self._refresh_failures_label()

    def _request_stop(self):
        if hasattr(self, "worker"):
            self.worker.stop()
        self.stop_segmentation_btn.setEnabled(False)
        self.stop_segmentation_btn.setText("Stopping after current sample…")

    def _continue_segmentation(self):
        paths = getattr(self, "_continuation_sample_paths", [])
        if paths:
            self.start_segmentation(
                sample_override=paths,
                index_offset=getattr(self, "_continuation_index_offset", 0),
                total_count=getattr(self, "_continuation_total_count", len(paths)),
            )

    def _on_sample_failed(self, sample_path, last_error_line):
        """A single sample errored; the run continues with the next one."""
        if not hasattr(self, "_failed_samples"):
            self._failed_samples = []
        name = os.path.basename(os.path.normpath(sample_path)) or sample_path
        self._failed_samples.append((name, last_error_line))
        self._refresh_failures_label()

    def _refresh_failures_label(self):
        """Show the skipped-sample count plus the most recent error line."""
        failures = getattr(self, "_failed_samples", [])
        if not failures:
            self.segmentation_failures_label.setVisible(False)
            return
        name, last_error_line = failures[-1]
        count = len(failures)
        prefix = (
            "1 sample was skipped"
            if count == 1
            else f"{count} samples were skipped"
        )
        text = f"⚠ {prefix} due to errors. Last failure — {name}: {last_error_line}"
        if _is_key_error(last_error_line):
            text += f"\n{CHANNEL_NAME_HINT}"
        self.segmentation_failures_label.setText(text)
        tooltip = "\n".join(f"{n}: {err}" for n, err in failures)
        if any(_is_key_error(err) for _, err in failures):
            tooltip += f"\n\n{CHANNEL_NAME_HINT}"
        self.segmentation_failures_label.setToolTip(tooltip)
        self.segmentation_failures_label.setVisible(True)

    def _on_worker_heartbeat(self, msg):
        import time
        self._last_heartbeat_msg = msg
        self._last_heartbeat_time = time.time()
        self.segmentation_summary_label.setStyleSheet("")
        self.segmentation_summary_label.setText(msg)
        self.segmentation_summary_label.setVisible(True)

    def _watchdog_tick(self):
        import time
        if not getattr(self, "_segmentation_running", False):
            return
        if hasattr(self, "worker") and not self.worker.isRunning():
            self._watchdog_timer.stop()
            self._segmentation_running = False
            self.run_segmentation_btn.setEnabled(True)
            self.stop_segmentation_btn.setVisible(False)
            self.continue_segmentation_btn.setVisible(False)
            self.progressbar.setVisible(False)
            self.segmentation_summary_label.setStyleSheet("color: #FF4444;")
            self.segmentation_summary_label.setText(
                "Error: the segmentation process stopped unexpectedly. Check the terminal output for details."
            )
            self.segmentation_summary_label.setVisible(True)
            self._refresh_failures_label()
            return
        elapsed = int(time.time() - self._last_heartbeat_time)
        minutes, seconds = divmod(elapsed, 60)
        self.segmentation_summary_label.setStyleSheet("")
        self.segmentation_summary_label.setText(
            f"{self._last_heartbeat_msg} ({minutes}m {seconds}s elapsed on this sample)"
        )
        self.segmentation_summary_label.setVisible(True)

    def _create_config_buttons_layout(self):
        layout = QHBoxLayout()
        layout.setSpacing(8)
        save_btn = QPushButton("Save configuration")
        save_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        save_btn.clicked.connect(self.save_configuration)
        load_btn = QPushButton("Load configuration")
        load_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        load_btn.clicked.connect(self.load_configuration)
        layout.addWidget(save_btn)
        layout.addWidget(load_btn)
        layout.addStretch()
        return layout

    def save_configuration(self):
        import json
        config = {
            "model_name": self.model_combo_box.currentText(),
            "model_path": self.model_combo_box.currentData() or "",
            "channel_count": self.channel_count,
            "channels": [
                {"name": le.text(), "type": cb.currentText()}
                for _, (_, cb, le) in sorted(self.channel_widgets.items())
            ],
            "crop_mode": self._crop_mode(),
            "manual_crop_fixed": self.manual_crop_fixed_checkbox.isChecked(),
            "save_crop_as": self.save_crop_as.currentText(),
            "run_mode": self.run_mode_combo.currentText(),
            "breaking_threshold": self.breaking_threshold_input.text(),
            "size_2d_filter_multiplier": self.size_2d_filter_input.text(),
            "voxel_z": self.voxel_size_z_input.text(),
            "voxel_x": self.voxel_size_x_input.text(),
            "voxel_y": self.voxel_size_y_input.text(),
            "save_frames": self.save_frames_checkbox.isChecked(),
            "save_segmentation": self.save_segmentation_checkbox.isChecked(),
            "do_phenotype_calling": self.do_phenotype_calling_checkbox.isChecked(),
            "create_split_phenotype_mask": self.create_split_phenotype_mask.isChecked(),
            "phenotype_1": self.phenotype_1.currentText(),
            "phenotype_2": self.phenotype_2.currentText(),
            "cutoff_method": self.calculate_cutoff.currentText(),
            "custom_cutoff": self.custom_cutoff_input.text(),
            "raw_or_background_subtracted": self.raw_or_background_subtracted.currentText(),
            "phenotype_1_region": self.phenotype_1_region.currentData(),
            "phenotype_2_region": self.phenotype_2_region.currentData(),
            "measure_regions": self.get_measure_regions(),
            "cytoplasm_size": self.cytoplasm_size_input.text(),
            "save_measurement_mask": self.save_measurement_mask_checkbox.isChecked(),
            "calculate_neighbour_statistics": self.calculate_neighbour_statistics_checkbox.isChecked(),
            "use_knn": self.calculate_neighbours_knn_checkbox.isChecked(),
            "knn": self.knn_input.text(),
            "phenotype_similarity_knn": self.calculate_phenotype_similarity_knn_checkbox.isChecked(),
            "use_touching": self.calculate_neighbours_touching_checkbox.isChecked(),
            "touching_dilation_um": self.touching_dilation_um_input.text(),
            "phenotype_similarity_touching": self.calculate_phenotype_similarity_touching_checkbox.isChecked(),
            "advanced_statistics": [
                self.calculate_advanced_statistics_list.item(i).text()
                for i in range(self.calculate_advanced_statistics_list.count())
                if self.calculate_advanced_statistics_list.item(i).isSelected()
            ],
        }
        path, _ = QFileDialog.getSaveFileName(
            self, "Save configuration", "", "JSON (*.json)"
        )
        if path:
            with open(path, "w") as f:
                json.dump(config, f, indent=2)

    def load_configuration(self):
        import json
        path, _ = QFileDialog.getOpenFileName(
            self, "Load configuration", "", "JSON (*.json)"
        )
        if not path:
            return
        try:
            with open(path) as f:
                config = json.load(f)
        except Exception as e:
            self.segmentation_summary_label.setStyleSheet("color: #FF4444;")
            self.segmentation_summary_label.setText(f"Could not load configuration: {e}")
            self.segmentation_summary_label.setVisible(True)
            return

        # Restore channels
        while self.channel_count > 1:
            self.delete_channel()
        channels = config.get("channels", [])
        if channels:
            _, cb, le = self.channel_widgets[1]
            le.setText(channels[0].get("name", "Channel_1"))
            idx = cb.findText(channels[0].get("type", ""))
            if idx >= 0:
                cb.setCurrentIndex(idx)
            for ch in channels[1:]:
                self.add_channel()
                _, cb, le = self.channel_widgets[self.channel_count]
                le.setText(ch.get("name", f"Channel_{self.channel_count}"))
                idx = cb.findText(ch.get("type", ""))
                if idx >= 0:
                    cb.setCurrentIndex(idx)
        self._update_phenotype_combos()

        # Restore model selection — try saved path first, then name, else keep default
        saved_model_path = config.get("model_path", "")
        saved_model_name = config.get("model_name", "")
        self._refresh_model_options()
        selected = False
        if saved_model_path and os.path.exists(saved_model_path):
            idx = self.model_combo_box.findText(saved_model_name)
            if idx >= 0:
                self.model_combo_box.setCurrentIndex(idx)
                selected = True
        if not selected and saved_model_name:
            idx = self.model_combo_box.findText(saved_model_name)
            if idx >= 0:
                self.model_combo_box.setCurrentIndex(idx)

        def _set_check(widget, key, default=False):
            widget.setChecked(bool(config.get(key, default)))

        def _set_combo(widget, key, default=""):
            idx = widget.findText(str(config.get(key, default)))
            if idx >= 0:
                widget.setCurrentIndex(idx)

        crop_mode = config.get("crop_mode")
        if crop_mode is None:
            # Configs from before crop_mode stored crop/skip checkboxes. Crop without skip
            # maps to "crop", not "recrop", so loading a config never overwrites crops.
            crop_mode = "crop" if config.get("do_crop_sample", True) else "none"
        {
            "crop": self.crop_new_radio,
            "recrop": self.crop_again_radio,
            "none": self.no_crop_radio,
        }.get(crop_mode, self.crop_new_radio).setChecked(True)
        _set_check(self.manual_crop_fixed_checkbox, "manual_crop_fixed")
        _set_combo(self.save_crop_as, "save_crop_as", ".tif")
        _set_combo(self.run_mode_combo, "run_mode", "Full pipeline")
        self.breaking_threshold_input.setText(config.get("breaking_threshold", "2.5"))
        self.size_2d_filter_input.setText(config.get("size_2d_filter_multiplier", "15"))
        self.voxel_size_z_input.setText(config.get("voxel_z", ""))
        self.voxel_size_x_input.setText(config.get("voxel_x", ""))
        self.voxel_size_y_input.setText(config.get("voxel_y", ""))
        _set_check(self.save_frames_checkbox, "save_frames")
        _set_check(self.save_segmentation_checkbox, "save_segmentation")
        _set_check(self.do_phenotype_calling_checkbox, "do_phenotype_calling")
        _set_check(self.create_split_phenotype_mask, "create_split_phenotype_mask")
        _set_combo(self.phenotype_1, "phenotype_1")
        _set_combo(self.phenotype_2, "phenotype_2")
        _set_combo(self.calculate_cutoff, "cutoff_method")
        self.custom_cutoff_input.setText(config.get("custom_cutoff", "0.0"))
        _set_combo(self.raw_or_background_subtracted, "raw_or_background_subtracted")
        # Configs written before regions were multi-select carry a single label.
        legacy = config.get("measure_intensity_in", "Nuclei").lower().replace(" ", "_")
        regions = set(config.get("measure_regions") or [legacy])
        for tag, box in self.region_checkboxes.items():
            box.setChecked(tag == "nuclei" or tag in regions)
        for combo, key in (
            (self.phenotype_1_region, "phenotype_1_region"),
            (self.phenotype_2_region, "phenotype_2_region"),
        ):
            index = combo.findData(config.get(key, "nuclei"))
            combo.setCurrentIndex(index if index >= 0 else 0)
        self.cytoplasm_size_input.setText(config.get("cytoplasm_size", "5"))
        _set_check(self.save_measurement_mask_checkbox, "save_measurement_mask")
        _set_check(self.calculate_neighbour_statistics_checkbox, "calculate_neighbour_statistics")
        _set_check(self.calculate_neighbours_knn_checkbox, "use_knn")
        self.knn_input.setText(config.get("knn", ""))
        _set_check(self.calculate_phenotype_similarity_knn_checkbox, "phenotype_similarity_knn")
        _set_check(self.calculate_neighbours_touching_checkbox, "use_touching")
        self.touching_dilation_um_input.setText(config.get("touching_dilation_um", "1.0"))
        _set_check(self.calculate_phenotype_similarity_touching_checkbox, "phenotype_similarity_touching")

        selected = set(config.get("advanced_statistics", []))
        for i in range(self.calculate_advanced_statistics_list.count()):
            item = self.calculate_advanced_statistics_list.item(i)
            item.setSelected(item.text() in selected)

        # Re-trigger toggle slots so dependent sub-widgets show/hide correctly
        self._region_selection_changed()
        self._on_crop_mode_changed()
        self._toggle_phenotype_daughters(self.do_phenotype_calling_checkbox.isChecked())
        self._toggle_neighbour_statistics(self.calculate_neighbour_statistics_checkbox.isChecked())
        self._toggle_knn_neighbour_options(self.calculate_neighbours_knn_checkbox.isChecked())
        self._toggle_touching_neighbour_options(self.calculate_neighbours_touching_checkbox.isChecked())
        self._toggle_custom_cutoff(self.calculate_cutoff.currentText())

    def _setup_view_data_widgets(self):
        """Setup all widgets for View Data tab"""
        # Sample selection section
        sample_section = QVBoxLayout()
        sample_section.setSpacing(5)
        select_label = QLabel("Select samples:")
        select_label.setStyleSheet("font-weight: bold;")
        sample_section.addWidget(select_label)
        self.view_data_sample_list = self._create_single_sample_list()
        sample_section.addWidget(self.view_data_sample_list)
        self.view_data_layout.addLayout(sample_section, 1)

        # Analysis Options section
        analysis_section = QVBoxLayout()
        analysis_section.setSpacing(5)
        napari_label = QLabel("Which channels to show in Napari:")
        napari_label.setStyleSheet("font-weight: bold;")
        analysis_section.addWidget(napari_label)

        # Channel selector
        channel_layout = self._create_channel_viewing_menu()
        analysis_section.addLayout(channel_layout)

        # Show segmentation checkbox
        self.show_segmentation_checkbox = QCheckBox("Show segmentation mask")
        self.show_segmentation_checkbox.setChecked(True)
        analysis_section.addWidget(self.show_segmentation_checkbox)

        # Whole-cell and cytoplasm masks only exist when the analysis step was
        # run with "save measurement mask", so the boxes stay hidden until a
        # selected sample actually has one.
        self.region_mask_checkboxes = {}
        for tag, label in self._REGION_MASKS:
            checkbox = QCheckBox(label)
            checkbox.setChecked(False)
            checkbox.setVisible(False)
            analysis_section.addWidget(checkbox)
            self.region_mask_checkboxes[tag] = checkbox

        self.show_phenotype_segmentations_checkbox = QCheckBox("Show split phenotype segmentations")
        self.show_phenotype_segmentations_checkbox.setChecked(False)
        self.show_phenotype_segmentations_checkbox.toggled.connect(
            lambda checked: self.phenotype_color_widget.setVisible(checked)
        )
        analysis_section.addWidget(self.show_phenotype_segmentations_checkbox)

        self._phenotype_color_combos = {}
        self.phenotype_color_widget = QWidget()
        pheno_color_vlayout = QVBoxLayout()
        pheno_color_vlayout.setContentsMargins(20, 0, 0, 0)
        pheno_color_vlayout.setSpacing(4)
        self.phenotype_color_widget.setLayout(pheno_color_vlayout)
        self.phenotype_color_widget.setVisible(False)
        analysis_section.addWidget(self.phenotype_color_widget)

        view_list = self._get_list_widget(self.view_data_sample_list)
        view_list.itemSelectionChanged.connect(self._update_phenotype_color_rows)
        view_list.itemSelectionChanged.connect(self._update_region_mask_checkboxes)

        self.view_data_layout.addLayout(analysis_section)

        # Buttons section
        button_section = QVBoxLayout()
        button_section.setSpacing(5)
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        self.view_button = QPushButton("View selected samples in Napari")
        self.view_button.clicked.connect(self.view_selected_samples)
        button_layout.addWidget(self.view_button)
        button_section.addLayout(button_layout)
        self.view_status_label = QLabel("")
        self.view_status_label.setVisible(False)
        button_section.addWidget(self.view_status_label)
        self.view_data_layout.addLayout(button_section)

    # Per-cell masks add_advanced_statistics can write beside the segmentation,
    # as "{sample}_segmented_{tag}.tif".
    _REGION_MASKS = (
        ("whole_cell", "Show whole-cell segmentation"),
        ("cytoplasm", "Show cytoplasm segmentation"),
    )

    def region_mask_path(self, base_path, sample, tag):
        return os.path.join(base_path, sample, f"{sample}_segmented_{tag}.tif")

    def _update_region_mask_checkboxes(self):
        """Offer a mask only when a selected sample actually has that file."""
        list_widget = self._get_list_widget(self.view_data_sample_list)
        selected = [item.text() for item in list_widget.selectedItems()]
        base_path = self.path_text.text()

        # Tracked here rather than read back from the widgets: a checkbox
        # reports isVisible() False whenever its window is not on screen yet.
        self._available_region_masks = set()
        for tag, checkbox in self.region_mask_checkboxes.items():
            available = any(
                os.path.exists(self.region_mask_path(base_path, sample, tag))
                for sample in selected
            )
            checkbox.setVisible(available)
            if available:
                self._available_region_masks.add(tag)
            else:
                checkbox.setChecked(False)

    def selected_region_masks(self):
        """Tags the user ticked, among those that are actually available."""
        available = getattr(self, "_available_region_masks", set())
        return [
            tag
            for tag, checkbox in self.region_mask_checkboxes.items()
            if tag in available and checkbox.isChecked()
        ]

    _PHENOTYPE_COLORS = ["Gray", "Red", "Green", "Blue", "Cyan", "Magenta", "Yellow", "White"]
    _PHENOTYPE_COLOR_MAP = {
        "Gray": "gray", "Red": "red", "Green": "green", "Blue": "blue",
        "Cyan": "cyan", "Magenta": "magenta", "Yellow": "yellow", "White": "white",
    }

    def _update_phenotype_color_rows(self):
        list_widget = self._get_list_widget(self.view_data_sample_list)
        selected = [item.text() for item in list_widget.selectedItems()]

        seen = set()
        phenotype_names = []
        for sample in selected:
            sample_dir = os.path.join(self.path_text.text(), sample)
            if not os.path.isdir(sample_dir):
                continue
            for fname in sorted(os.listdir(sample_dir)):
                if fname.startswith(f"{sample}_") and fname.endswith("_mask.tif"):
                    name = fname[len(sample) + 1 : -len("_mask.tif")]
                    if name not in seen:
                        phenotype_names.append(name)
                        seen.add(name)

        layout = self.phenotype_color_widget.layout()
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._phenotype_color_combos = {}
        for name in phenotype_names:
            row_widget = QWidget()
            row_layout = QHBoxLayout()
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.addWidget(QLabel(name))
            combo = QComboBox()
            combo.addItems(self._PHENOTYPE_COLORS)
            row_layout.addWidget(combo)
            row_widget.setLayout(row_layout)
            layout.addWidget(row_widget)
            self._phenotype_color_combos[name] = combo

    def _create_channel_viewing_menu(self):
        """Add UI elements for channel viewing here"""
        layout = QVBoxLayout()
        layout.setSpacing(10)

        # Use grid layout for aligned columns
        self.channel_grid_viewing = QGridLayout()
        self.channel_grid_viewing.setSpacing(10)

        # Header row
        for text, col in [("Channel", 0), ("Color", 1), ("Name", 2)]:
            lbl = QLabel(text)
            lbl.setStyleSheet("font-weight: bold;")
            self.channel_grid_viewing.addWidget(lbl, 0, col)

        # Initial channel line
        self.channel_count_viewing = 1
        self.channel_widgets_viewing = {}  # Store widgets for deletion
        self._add_channel_viewing_grid_row(self.channel_count_viewing)

        layout.addLayout(self.channel_grid_viewing)

        # Add/Delete channel buttons
        button_layout = QHBoxLayout()
        add_channel_btn = QPushButton("Add channel")
        add_channel_btn.clicked.connect(self.add_channel_viewing)
        delete_channel_btn = QPushButton("Delete channel")
        delete_channel_btn.clicked.connect(self.delete_channel_viewing)
        button_layout.addWidget(add_channel_btn)
        button_layout.addWidget(delete_channel_btn)
        layout.addLayout(button_layout)

        return layout

    def _add_channel_viewing_grid_row(self, channel_number):
        """Add a new row to the channel grid"""
        row = self.channel_grid_viewing.rowCount()

        # Channel number label
        label = QLabel(f"{channel_number}:")
        self.channel_grid_viewing.addWidget(label, row, 0)

        # Color combobox
        combo_box = QComboBox()
        color_items = {
            "Blue": "#4488FF",
            "Cyan": "#00FFFF",
            "Green": "#00EE00",
            "Yellow": "#FFFF00",
            "Red": "#FF4444",
            "Magenta": "#FF44FF",
            "Gray": "#AAAAAA",
            "Bop Orange": "#FF6D00",
            "Bop Purple": "#AA44FF",
            "Bop Blue": "#00AAFF",
        }
        color_list = list(color_items.values())
        for name, hex_color in color_items.items():
            combo_box.addItem(name)
            combo_box.model().item(combo_box.count() - 1).setForeground(
                QBrush(QColor(hex_color))
            )

        def _update_combo_color(index, cb=combo_box, colors=color_list):
            cb.setStyleSheet(f"QComboBox {{ color: {colors[index]}; }}")

        combo_box.currentIndexChanged.connect(_update_combo_color)
        _update_combo_color(0)  # set initial color
        self.channel_grid_viewing.addWidget(combo_box, row, 1)

        # Channel name input
        line_edit = QLineEdit(f"Channel_{channel_number}")
        self.channel_grid_viewing.addWidget(line_edit, row, 2)

        # Store widgets for later deletion
        self.channel_widgets_viewing[channel_number] = (label, combo_box, line_edit)

    def add_channel_viewing(self):
        """Add a new channel viewing line"""
        self.channel_count_viewing += 1
        self._add_channel_viewing_grid_row(self.channel_count_viewing)

    def delete_channel_viewing(self):
        """Delete the last channel viewing line"""
        if self.channel_count_viewing > 1:
            # Delete widgets from grid
            label, combo_box, line_edit = self.channel_widgets_viewing[
                self.channel_count_viewing
            ]
            label.deleteLater()
            combo_box.deleteLater()
            line_edit.deleteLater()

            # Remove from storage
            del self.channel_widgets_viewing[self.channel_count_viewing]
            self.channel_count_viewing -= 1

    def get_channel_settings_viewing(self):
        """Extract channel names and types from the grid"""
        channel_names = []
        channel_colors = []
        for channel_num in sorted(self.channel_widgets_viewing.keys()):
            _, combo_box, line_edit = self.channel_widgets_viewing[channel_num]
            channel_colors.append(combo_box.currentText())
            channel_names.append(line_edit.text())
        return channel_names, channel_colors

    def view_selected_samples(self):
        list_widget = self._get_list_widget(self.view_data_sample_list)
        selected = [item.text() for item in list_widget.selectedItems()]
        if not selected:
            return
        show_segmentation = self.show_segmentation_checkbox.isChecked()
        show_phenotype_segmentations = self.show_phenotype_segmentations_checkbox.isChecked()
        phenotype_colors = {
            name: self._PHENOTYPE_COLOR_MAP[combo.currentText()]
            for name, combo in self._phenotype_color_combos.items()
        }
        channel_names, channel_colors = self.get_channel_settings_viewing()

        self.view_button.setEnabled(False)
        self.view_status_label.setText("Loading data...")
        self.view_status_label.setVisible(True)

        self.viewer_worker = ViewerWorker(
            selected,
            self.path_text.text(),
            channel_names,
            channel_colors,
            show_segmentation,
            show_phenotype_segmentations,
            phenotype_colors,
            self.selected_region_masks(),
        )
        self.viewer_worker.data_loaded.connect(self._launch_napari_viewer)
        self.viewer_worker.error_occurred.connect(self._viewer_error)
        self.viewer_worker.start()

    def _contrast_from_sample(self, channel_stack):
        """Contrast limits and slider range for one channel of a (T, Z, Y, X) movie.

        napari works these out itself, but on a lazy movie it only inspects a
        single plane, which on a mostly-background plane collapses the range and
        saturates the image. Reading one timepoint here is both correct and
        cheap enough to keep the movie lazy.

        Returns (contrast_limits, slider_range).
        """
        import numpy as np

        from utils.load_image import as_numpy

        timepoint = as_numpy(channel_stack[len(channel_stack) // 2])
        # Thin out Z so a tall stack does not make the percentiles expensive.
        step = max(1, timepoint.shape[0] // 8) if timepoint.ndim >= 3 else 1
        sample = np.ravel(timepoint[::step])

        data_max = float(sample.max()) if sample.size else 1.0

        # Cropped movies are zeroed outside the organoid; those pixels are not
        # background and would drag the black point down to 0.
        signal = sample[sample > 0]
        if signal.size == 0:
            return (0.0, max(data_max, 1.0)), (0.0, max(data_max, 1.0))

        background = float(np.median(signal))
        foreground = float(np.percentile(signal, 99.95))

        return (background, foreground), (0.0, max(data_max, foreground))

    def _launch_napari_viewer(self, data):
        import napari
        import napari.viewer  # force lazy-loaded submodule to resolve before shiboken2 interferes

        self.view_button.setEnabled(True)
        self.view_status_label.setVisible(False)

        import numpy as np

        channel_names = data["channel_names"]
        channel_colors = data["channel_colors"]
        show_segmentation = data["show_segmentation"]
        phenotype_colors = data.get("phenotype_colors", {})

        viewer = napari.Viewer(ndisplay=3)
        for sample_data in data["samples_data"]:
            sample = sample_data["sample"]
            movie = sample_data["movie"]
            voxel_size = sample_data["voxel_size"]
            segmentation = sample_data["segmentation"]
            properties_path = sample_data.get("properties_path")
            seg_metadata = {"sample": sample, "voxel_size": voxel_size}
            if properties_path:
                seg_metadata["properties_path"] = properties_path

            for i, channel_name in enumerate(channel_names):
                channel_stack = movie[:, i]
                limits, value_range = self._contrast_from_sample(channel_stack)
                layer = viewer.add_image(
                    channel_stack,
                    name=channel_name,
                    colormap=channel_colors[i].lower(),
                    visible=True,
                    blending="additive",
                    scale=voxel_size,
                    contrast_limits=limits,
                )
                # Slider spans the full intensity range; the handles start at
                # background..foreground. Limits are re-applied because widening
                # the range can reset them.
                layer.contrast_limits_range = value_range
                layer.contrast_limits = limits
            if show_segmentation and segmentation is not None:
                if len(segmentation.shape) > 4:
                    for j in range(segmentation.shape[1]):
                        viewer.add_labels(
                            segmentation[:, j],
                            name=f"{sample} segmentation_{j}",
                            visible=True,
                            blending="additive",
                            rendering="iso_categorical",
                            scale=voxel_size,
                            iso_gradient_mode="smooth",
                            metadata=dict(seg_metadata),
                        )
                else:
                    viewer.add_labels(
                        segmentation,
                        name=f"{sample} segmentation",
                        visible=True,
                        blending="additive",
                        rendering="iso_categorical",
                        scale=voxel_size,
                        iso_gradient_mode="smooth",
                        metadata=dict(seg_metadata),
                    )

            # Whole-cell / cytoplasm masks carry the same cell labels as the
            # segmentation, so they get its metadata too and the property
            # filter works on them as well.
            for region in sample_data.get("region_masks", []):
                viewer.add_labels(
                    region["data"],
                    name=f"{sample} {region['name']} segmentation",
                    visible=True,
                    blending="additive",
                    rendering="iso_categorical",
                    scale=voxel_size,
                    iso_gradient_mode="smooth",
                    metadata=dict(seg_metadata),
                )

            for pheno in sample_data.get("phenotype_masks", []):
                mask = pheno["data"]
                pheno_name = pheno["name"]
                viewer.add_labels(
                    mask,
                    name=f"{sample} {pheno_name} mask",
                    visible=True,
                    blending="additive",
                    rendering="iso_categorical",
                    scale=voxel_size,
                    iso_gradient_mode="smooth",
                )
                binary = (mask > 0).astype(np.uint8)
                binary_color = phenotype_colors.get(pheno_name, "gray")
                viewer.add_labels(
                    binary,
                    name=f"{sample} {pheno_name} binary",
                    visible=True,
                    blending="additive",
                    rendering="iso_categorical",
                    scale=voxel_size,
                    iso_gradient_mode="smooth",
                    colormap={0: "", 1: binary_color},
                )

        # Add the property-based label filtering widget (slider plugin).
        try:
            from utils.napari_slider_plugin import attach_property_filter

            attach_property_filter(viewer)
        except Exception as e:
            print(f"Could not attach property filter widget: {e}")

        self._release_files_when_closed(viewer)

    def _release_files_when_closed(self, viewer):
        """Drop the layers when the viewer window goes away.

        Layers hold the lazily-loaded arrays, which hold the open source files.
        Until they are released the sample cannot be moved, renamed or deleted
        on Windows, so closing the viewer has to actually let go.
        """
        def release(*_):
            try:
                viewer.layers.clear()
            except Exception:
                pass
            gc.collect()

        try:
            viewer.window._qt_window.destroyed.connect(release)
        except Exception as error:
            # Private napari API: if it moves, the files stay open until the
            # objects are collected anyway, so this is not worth failing over.
            print(f"Note: could not hook viewer teardown ({error}).")

    def _viewer_error(self, error_message):
        self.view_button.setEnabled(True)
        self.view_status_label.setText(f"Error: {error_message}")
        self.view_status_label.setVisible(True)

    def _setup_export_data_widgets(self):
        """Setup all widgets for Export Data tab"""
        # Sample selection section
        sample_section = QVBoxLayout()
        sample_section.setSpacing(5)
        select_label = QLabel("Select samples:")
        select_label.setStyleSheet("font-weight: bold;")
        sample_section.addWidget(select_label)
        self.export_data_sample_list = self._create_single_sample_list()
        sample_section.addWidget(self.export_data_sample_list)
        self.export_data_layout.addLayout(sample_section)

        # Buttons
        self.export_button = QPushButton("Export selected samples to TSV")
        self.export_button.clicked.connect(self.create_export_data)
        self.export_data_layout.addWidget(self.export_button)

        self.export_summary_button = QPushButton("Export summary of selected samples to TSV")
        self.export_summary_button.clicked.connect(self.create_export_summary)
        self.export_data_layout.addWidget(self.export_summary_button)

        # self.show_plots_button = QPushButton("Show Plots")
        # self.show_plots_button.clicked.connect(self.show_plots)
        # self.export_data_layout.addWidget(self.show_plots_button)

        self.plot_status_label = QLabel("")
        self.plot_status_label.setVisible(False)
        self.export_data_layout.addWidget(self.plot_status_label)

        # Scrollable plot area
        self.plot_scroll_area = QScrollArea()
        self.plot_scroll_area.setWidgetResizable(True)
        self.plot_scroll_area.setVisible(False)
        self.plot_container = QWidget()
        self.plot_container_layout = QVBoxLayout()
        self.plot_container_layout.setSpacing(10)
        self.plot_container.setLayout(self.plot_container_layout)
        self.plot_scroll_area.setWidget(self.plot_container)
        self.export_data_layout.addWidget(self.plot_scroll_area, 1)

    def create_export_data(self):
        list_widget = self._get_list_widget(self.export_data_sample_list)
        selected_samples = [item.text() for item in list_widget.selectedItems()]
        if not selected_samples:
            return
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Exported Data",
            "",
            "TSV Files (*.tsv);;All Files (*)",
        )
        if not save_path:
            return
        self.export_button.setEnabled(False)
        self.export_worker = ExportWorker(
            selected_samples, self.path_text.text(), save_path
        )
        self.export_worker.finished.connect(lambda: self.export_button.setEnabled(True))
        self.export_worker.error_occurred.connect(self._export_error)
        self.export_worker.start()

    def create_export_summary(self):
        list_widget = self._get_list_widget(self.export_data_sample_list)
        selected_samples = [item.text() for item in list_widget.selectedItems()]
        if not selected_samples:
            return
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Exported Summary",
            "",
            "TSV Files (*.tsv);;All Files (*)",
        )
        if not save_path:
            return
        self.export_summary_button.setEnabled(False)
        self.export_worker = ExportSummaryWorker(
            selected_samples, self.path_text.text(), save_path
        )
        self.export_worker.finished.connect(lambda: self.export_summary_button.setEnabled(True))
        self.export_worker.error_occurred.connect(self._export_summary_error)
        self.export_worker.start()

    def _export_error(self, error_message):
        self.export_button.setEnabled(True)
        self.plot_status_label.setText(f"Export error: {error_message}")
        self.plot_status_label.setVisible(True)

    def _export_summary_error(self, error_message):
        self.export_summary_button.setEnabled(True)
        self.plot_status_label.setText(f"Export error: {error_message}")
        self.plot_status_label.setVisible(True)

    def show_plots(self):
        list_widget = self._get_list_widget(self.export_data_sample_list)
        selected_samples = [item.text() for item in list_widget.selectedItems()]
        if not selected_samples:
            return

        self.show_plots_button.setEnabled(False)
        self.plot_status_label.setText("Loading data...")
        self.plot_status_label.setVisible(True)

        self.plot_worker = PlotWorker(selected_samples, self.path_text.text())
        self.plot_worker.data_loaded.connect(self._render_plots)
        self.plot_worker.error_occurred.connect(self._plot_error)
        self.plot_worker.start()

    def _render_plots(self, full_results):
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

        if get_windows_theme():
            plt.style.use("dark_background")

        self.show_plots_button.setEnabled(True)
        self.plot_status_label.setVisible(False)

        if full_results.empty:
            return

        self._clear_layout(self.plot_container_layout)
        self.plot_scroll_area.setVisible(True)

        def add_figure(fig):
            canvas = FigureCanvasQTAgg(fig)
            canvas.setStyleSheet("background: transparent;")
            canvas.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
            canvas.setMinimumHeight(400)
            self.plot_container_layout.addWidget(canvas)
            plt.close(fig)

        # --- Add your plots below ---
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor("none")
        ax.set_facecolor("none")

        cell_counts = (
            full_results.groupby(["sample", "frame"])
            .size()
            .reset_index(name="cell_count")
        )
        sns.pointplot(
            data=cell_counts,
            x="frame",
            y="cell_count",
            hue="sample",
            ax=ax,
        )

        legend = ax.get_legend()
        if legend:
            legend.set_frame_on(False)
            legend.remove()
        ax.spines[["right", "top"]].set_visible(False)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Cell count")
        ax.set_title("Cell count over time")
        ax.legend(
            frameon=False, bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0
        )
        fig.tight_layout()
        add_figure(fig)

    def _plot_error(self, error_message):
        self.show_plots_button.setEnabled(True)
        self.plot_status_label.setText(f"Error: {error_message}")
        self.plot_status_label.setVisible(True)


class ExportWorker(QThread):
    finished = Signal()
    error_occurred = Signal(str)

    def __init__(self, selected_samples, base_path, save_path):
        super().__init__()
        self.selected_samples = selected_samples
        self.base_path = base_path
        self.save_path = save_path

    def run(self):
        try:
            import pandas as pd

            data = []
            for sample in self.selected_samples:
                prop_path = os.path.join(
                    self.base_path, sample, f"{sample}_properties.tsv"
                )
                if os.path.exists(prop_path):
                    data.append(pd.read_csv(prop_path, sep="\t"))
            if data:
                pd.concat(data, ignore_index=True).to_csv(
                    self.save_path, sep="\t", index=False
                )
            self.finished.emit()
        except Exception as e:
            self.error_occurred.emit(str(e))

class ExportSummaryWorker(QThread):
    finished = Signal()
    error_occurred = Signal(str)

    def __init__(self, selected_samples, base_path, save_path):
        super().__init__()
        self.selected_samples = selected_samples
        self.base_path = base_path
        self.save_path = save_path

    def run(self):
        try:
            import pandas as pd

            data = []
            for sample in self.selected_samples:
                prop_path = os.path.join(
                    self.base_path, sample, f"{sample}_properties.tsv"
                )
                if os.path.exists(prop_path):
                    data.append(pd.read_csv(prop_path, sep="\t"))
            if data:
                data = pd.concat(data, ignore_index=True)
                time_col = next(
                    (col for col in ("time", "timepoint") if col in data.columns and data[col].nunique() > 1),
                    None,
                )
                group_cols = ["sample"] + ([time_col] if time_col else [])
                summary = (
                    data.groupby(group_cols)
                    .agg(cell_count=("label", "nunique"))
                    .reset_index()
                )
                for pheno_col in [c for c in data.columns if c.startswith("phenotype_")]:
                    expected = pheno_col[len("phenotype_"):].split("_vs_")
                    counts = (
                        data.groupby(group_cols + [pheno_col])
                        .size()
                        .unstack(fill_value=0)
                        .reindex(columns=expected, fill_value=0)
                        .rename(columns=lambda v: f"phenotype_{v}")
                        .reset_index()
                    )
                    summary = summary.merge(counts, on=group_cols, how="left")
                summary.to_csv(
                    self.save_path, sep="\t", index=False
                )

            self.finished.emit()
        except Exception as e:
            self.error_occurred.emit(str(e))


class PlotWorker(QThread):
    data_loaded = Signal(object)
    error_occurred = Signal(str)

    def __init__(self, selected_samples, base_path):
        super().__init__()
        self.selected_samples = selected_samples
        self.base_path = base_path

    def run(self):
        try:
            import pandas as pd

            frames = []
            for sample in self.selected_samples:
                prop_path = os.path.join(
                    self.base_path, sample, f"{sample}_properties.tsv"
                )
                if os.path.exists(prop_path):
                    frames.append(pd.read_csv(prop_path, sep="\t"))
            result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            self.data_loaded.emit(result)
        except Exception as e:
            self.error_occurred.emit(str(e))


class ViewerWorker(QThread):
    data_loaded = Signal(object)
    error_occurred = Signal(str)

    def __init__(
        self, selected, base_path, channel_names, channel_colors, show_segmentation,
        show_phenotype_segmentations=False, phenotype_colors=None,
        show_region_masks=()
    ):
        super().__init__()
        self.selected = selected
        self.base_path = base_path
        self.channel_names = channel_names
        self.channel_colors = channel_colors
        self.show_segmentation = show_segmentation
        self.show_phenotype_segmentations = show_phenotype_segmentations
        self.phenotype_colors = phenotype_colors or {}
        self.show_region_masks = list(show_region_masks or ())

    def run(self):
        try:
            import numpy as np

            samples_data = []
            for sample in self.selected:

                def _file_priority(f):
                    lowered = f.lower()
                    if lowered.endswith("_cropped.ims"):
                        return 0
                    if lowered.endswith("_cropped.tif"):
                        return 1
                    if lowered.endswith(".ims"):
                        return 2
                    return 3  # any other readable format

                input_files = sorted(
                    [
                        f
                        for f in os.listdir(os.path.join(self.base_path, sample))
                        if is_supported(f)
                    ],
                    key=_file_priority,
                )

                input_file = input_files[0] if input_files else None

                print(input_file)

                # Lazy: napari reads slices on demand, so opening a sample in the
                # viewer never pulls the whole movie into RAM.
                movie, voxel_size, _, _ = load_image(
                    os.path.join(self.base_path, sample, input_file)
                )

                print(voxel_size)
                print(f"movie {movie.shape}")

                segmentation = None
                if self.show_segmentation:
                    seg_path = os.path.join(
                        self.base_path, sample, f"{sample}_segmented.tif"
                    )
                    if os.path.exists(seg_path):
                        segmentation = tifffile.imread(seg_path)
                print(
                    f"segmentation {segmentation.shape if segmentation is not None else 'None'}"
                )

                region_masks = []
                for tag in self.show_region_masks:
                    mask_path = os.path.join(
                        self.base_path, sample, f"{sample}_segmented_{tag}.tif"
                    )
                    if os.path.exists(mask_path):
                        region_masks.append(
                            {
                                "name": tag.replace("_", " "),
                                "data": tifffile.imread(mask_path),
                            }
                        )
                        print(f"Loaded {tag} mask: {os.path.basename(mask_path)}")

                phenotype_masks = []
                if self.show_phenotype_segmentations:
                    sample_dir = os.path.join(self.base_path, sample)
                    for fname in sorted(os.listdir(sample_dir)):
                        if fname.startswith(f"{sample}_") and fname.endswith("_mask.tif"):
                            pheno_name = fname[len(sample) + 1 : -len("_mask.tif")]
                            mask = tifffile.imread(os.path.join(sample_dir, fname))
                            phenotype_masks.append({"name": pheno_name, "data": mask})
                            print(f"Loaded phenotype mask: {fname} ({pheno_name})")

                properties_path = os.path.join(
                    self.base_path, sample, f"{sample}_properties.tsv"
                )

                samples_data.append(
                    {
                        "sample": sample,
                        "movie": movie,
                        "voxel_size": voxel_size,
                        "segmentation": segmentation,
                        "region_masks": region_masks,
                        "phenotype_masks": phenotype_masks,
                        "properties_path": (
                            properties_path
                            if os.path.exists(properties_path)
                            else None
                        ),
                    }
                )

            self.data_loaded.emit(
                {
                    "samples_data": samples_data,
                    "channel_names": self.channel_names,
                    "channel_colors": self.channel_colors,
                    "show_segmentation": self.show_segmentation,
                    "phenotype_colors": self.phenotype_colors,
                }
            )
        except Exception as e:
            import traceback

            traceback.print_exc()
            self.error_occurred.emit(str(e))


class PrewarmWorker(QThread):
    """Import the pipeline's heavy modules in the background at app launch.

    First-time import of cellpose (and, on the viewer path, napari) plus the
    pipeline entry modules costs ~6s warm and far more on a cold cache. Doing it
    here — while the user is still configuring — means the imports are already
    cached in ``sys.modules`` by the time Run/View is clicked, so those handlers
    start effectively instantly. Only importing happens here (no Qt widgets, no
    napari.Viewer), which is safe off the main thread; every import is guarded so
    a broken optional module can never crash startup.
    """

    # Heavy leaf modules plus pipeline entry points. Importing the entry points
    # pulls in the rest of each dependency tree, so the whole pipeline import
    # graph gets warmed without listing every module by hand.
    _TARGETS = (
        "cellpose.models",
        "napari",
        "napari.viewer",
        "utils.load_model",
        "main_functions.segment_organoid",
        "main_functions.calculate_phenotypes",
        "main_functions.crop_sample",
        "main_functions.add_advanced_statistics",
        "main_functions.add_phenotype_similarity",
        "main_functions.split_phenotype_mask",
    )

    # Emitted once torch is imported here, carrying torch.cuda.is_available().
    # Lets the GPU banner resolve off the UI-show critical path.
    gpu_available = Signal(bool)

    def run(self):
        import importlib
        import time

        start = time.perf_counter()

        # torch first: it is the heaviest single import and gates the GPU check.
        try:
            import torch

            self.gpu_available.emit(bool( torch.cuda.is_available() or torch.mps.is_available() ))
        except Exception as exc:
            print(f"[prewarm] torch import failed: {type(exc).__name__}: {exc}")
            self.gpu_available.emit(False)

        for module_name in self._TARGETS:
            try:
                importlib.import_module(module_name)
            except Exception as exc:
                # A missing/broken optional module must not affect startup.
                print(f"[prewarm] skipped {module_name}: {type(exc).__name__}: {exc}")
        print(f"[prewarm] pipeline modules ready ({time.perf_counter() - start:.1f}s)")


class SampleConversionWorker(QThread):
    """Build sample folders off the UI thread.

    Handles both jobs the Load Data tab can start: splitting a multiposition
    file into one sample per position, and merging a MetaMorph .nd acquisition
    into a single sample. Cancelling is cooperative: the converters poll
    should_stop between timepoints, drop the partially written file and return
    whatever they finished.
    """

    progress_updated = Signal(int)
    status_changed = Signal(str)
    finished_converting = Signal(object)
    error_occurred = Signal(str)

    def __init__(self, jobs):
        """jobs: (path, convert) pairs, convert(path, progress, should_stop)."""
        super().__init__()
        self.jobs = jobs
        self.written = {}
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def was_cancelled(self):
        return self._stop_requested

    def run(self):
        for number, (path, convert) in enumerate(self.jobs):
            if self._stop_requested:
                break

            def progress(fraction, label, _number=number):
                self.progress_updated.emit(
                    int(100 * (_number + fraction) / len(self.jobs))
                )
                self.status_changed.emit(label)

            try:
                self.written[path] = convert(
                    path, progress, lambda: self._stop_requested
                )
            except Exception as error:
                self.error_occurred.emit(
                    f"{os.path.basename(path)}: {type(error).__name__}: {error}"
                )
                return

        self.finished_converting.emit(self.written)


class SegmentationWorker(QThread):
    progress_updated = Signal(int)
    finished = Signal(float)
    stopped = Signal(float, int)
    error_occurred = Signal(str)
    heartbeat = Signal(str)
    sample_failed = Signal(str, str)  # sample path, last line of the traceback

    def __init__(
        self,
        sample_path_list,
        cell_model_path,
        channel_names,
        channel_types,
        breaking_threshold,
        size_2d_filter_multiplier,
        do_phenotype_calling,
        phenotype_calling_only,
        create_split_phenotype_mask,
        phenotype_1,
        phenotype_2,
        cutoff_method,
        custom_cutoff,
        raw_or_background_subtracted,
        do_crop_sample,
        manual_crop_fixed,
        crop_mode,
        save_crop_as,
        save_frames,
        save_segmentation,
        extra_props,
        advanced_statistics_only,
        measure_regions,
        cytoplasm_size,
        save_measurement_mask,
        user_voxel_size,
        calculate_neighbour_statistics=False,
        use_knn_neighbours=False,
        knn_list=None,
        calculate_phenotype_similarity_knn=False,
        use_touching_neighbours_3d=False,
        touching_dilation_um=None,
        calculate_phenotype_similarity_touching=False,
        manually_cropped_fixed_samples=None,
        index_offset=0,
        total_count=None,
        phenotype_1_region="nuclei",
        phenotype_2_region="nuclei",
    ):
        super().__init__()
        self._stop_requested = False
        self.sample_path_list = sample_path_list
        self.index_offset = index_offset
        self.total_count = total_count if total_count is not None else len(sample_path_list)
        self.cell_model_path = cell_model_path
        self.channel_names = channel_names
        self.channel_types = channel_types
        self.breaking_threshold = breaking_threshold
        self.size_2d_filter_multiplier = size_2d_filter_multiplier
        self.do_phenotype_calling = do_phenotype_calling
        self.phenotype_calling_only = phenotype_calling_only
        self.create_split_phenotype_mask = create_split_phenotype_mask
        self.phenotype_1 = phenotype_1
        self.phenotype_2 = phenotype_2
        self.phenotype_1_region = phenotype_1_region
        self.phenotype_2_region = phenotype_2_region
        self.cutoff_method = cutoff_method
        self.custom_cutoff = custom_cutoff
        self.raw_or_background_subtracted = raw_or_background_subtracted
        self.do_crop_sample = do_crop_sample
        self.manual_crop_fixed = manual_crop_fixed
        self.crop_mode = crop_mode
        self.save_crop_as = save_crop_as
        self.save_frames = save_frames
        self.save_segmentation = save_segmentation
        self.extra_props = extra_props
        self.advanced_statistics_only = advanced_statistics_only
        self.measure_regions = measure_regions
        self.cytoplasm_size = cytoplasm_size
        self.save_measurement_mask = save_measurement_mask
        self.user_voxel_size = user_voxel_size
        self.calculate_neighbour_statistics = calculate_neighbour_statistics
        self.use_knn_neighbours = use_knn_neighbours
        self.knn_list = knn_list
        self.calculate_phenotype_similarity_knn = calculate_phenotype_similarity_knn
        self.use_touching_neighbours_3d = use_touching_neighbours_3d
        self.touching_dilation_um = touching_dilation_um
        self.calculate_phenotype_similarity_touching = (
            calculate_phenotype_similarity_touching
        )
        self.manually_cropped_fixed_samples = {
            os.path.normcase(os.path.normpath(p))
            for p in (manually_cropped_fixed_samples or [])
        }

    def stop(self):
        self._stop_requested = True

    def _needs_crop(self, sample_path):
        # Samples cropped manually before the worker started are already done.
        if os.path.normcase(os.path.normpath(sample_path)) in self.manually_cropped_fixed_samples:
            return False
        return needs_crop(sample_path, self.crop_mode)

    def run(self):
        try:
            # These are pre-warmed at app launch (see PrewarmWorker); if that is
            # still running, the import lock makes the lines below wait, so print
            # first to avoid a silent gap before "Using CUDA for processing".
            print("Preparing pipeline (loading libraries and model)...")
            from main_functions.segment_organoid import segment_organoid
            from utils.load_model import load_model
            from main_functions.calculate_phenotypes import (
                calculate_phenotypes,
                phenotype_column_name,
            )
            from main_functions.crop_sample import crop_sample
            from main_functions.split_phenotype_mask import split_phenotype_mask
            from main_functions.add_advanced_statistics import add_advanced_statistics
            from main_functions.add_phenotype_similarity import add_phenotype_similarity

            def _format_param_token(value):
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    return str(value)
                if numeric_value.is_integer():
                    return str(int(numeric_value))
                return format(numeric_value, "g").replace(".", "p")

            start_time = datetime.now()
            loaded_cell_model = load_model(self.cell_model_path)

            needs_auto_crop = (
                not self.phenotype_calling_only
                and not self.advanced_statistics_only
                and any(self._needs_crop(sample_path) for sample_path in self.sample_path_list)
            )

            if needs_auto_crop:
                from sam2.build_sam import build_sam2_video_predictor # type: ignore

                model_path = os.path.join(
                    os.path.dirname(os.path.dirname(__file__)), "models"
                )
                organoid_model = build_sam2_video_predictor(
                    os.path.join(model_path, "sam2.1_hiera_s.yaml"),
                    os.path.join(model_path, "sam2.1_hiera_small.pt"),
                )
            else:
                organoid_model = None

            failed_samples = []
            stopped_at_idx = len(self.sample_path_list)
            for idx, i in enumerate(self.sample_path_list):
                if self._stop_requested:
                    stopped_at_idx = idx
                    break
                global_idx = self.index_offset + idx
                self.heartbeat.emit(
                    f"Processing sample {global_idx + 1} of {self.total_count}: {os.path.basename(i)}"
                )
                try:
                    if (
                        not self.phenotype_calling_only
                        and not self.advanced_statistics_only
                    ):
                        if self._needs_crop(i):
                            crop_sample(
                                i,
                                self.channel_types,
                                organoid_model,
                                manual_fixed=self.manual_crop_fixed,
                                save_as=self.save_crop_as,
                                user_voxel_size=self.user_voxel_size,
                            )
                            remove_stale_cropped_files(i, self.save_crop_as)
                        elif self.do_crop_sample:
                            print(f"Skipping crop for {i}: using its existing cropped file.")
                        segment_organoid(
                            i,
                            loaded_cell_model,
                            self.channel_types,
                            self.channel_names,
                            self.breaking_threshold,
                            self.size_2d_filter_multiplier,
                            self.do_crop_sample,
                            save_frames=self.save_frames,
                            save_segmentation=self.save_segmentation,
                            user_voxel_size=self.user_voxel_size,
                        )

                    # Statistics also measure the phenotype regions, so they run first.
                    stats = (
                        not self.phenotype_calling_only or self.advanced_statistics_only
                    )
                    regions = list(
                        dict.fromkeys(
                            (list(self.measure_regions) if stats else [])
                            + (
                                [self.phenotype_1_region, self.phenotype_2_region]
                                if self.do_phenotype_calling
                                else []
                            )
                        )
                    )
                    if stats or any(region != "nuclei" for region in regions):
                        print("Calculating advanced statistics...")
                        add_advanced_statistics(
                            i,
                            self.extra_props if stats else [],
                            self.channel_names,
                            measure_regions=regions,
                            cytoplasm_size=self.cytoplasm_size,
                            save_measurement_mask=self.save_measurement_mask,
                            user_voxel_size=self.user_voxel_size,
                            calculate_neighbour_statistics=stats
                            and self.calculate_neighbour_statistics,
                            use_knn_neighbours=self.use_knn_neighbours,
                            knn_list=self.knn_list,
                            use_touching_neighbours_3d=self.use_touching_neighbours_3d,
                            touching_dilation_um=self.touching_dilation_um,
                        )

                    phenotype_column = None
                    if self.do_phenotype_calling:
                        print("Calculating phenotypes...")
                        calculate_phenotypes(
                            i,
                            self.phenotype_1,
                            self.phenotype_2,
                            self.cutoff_method,
                            self.custom_cutoff,
                            self.raw_or_background_subtracted,
                            phenotype_1_region=self.phenotype_1_region,
                            phenotype_2_region=self.phenotype_2_region,
                        )
                        phenotype_column = phenotype_column_name(
                            self.phenotype_1,
                            self.phenotype_1_region,
                            self.phenotype_2,
                            self.phenotype_2_region,
                        )

                    if self.calculate_phenotype_similarity_knn:
                        if not self.use_knn_neighbours or self.knn_list is None:
                            print(
                                "Skipping KNN phenotype similarity score: KNN neighbours are disabled."
                            )
                        else:
                            print(
                                "Calculating phenotype similarity ratio based on KNN neighbours..."
                            )
                            for knn in self.knn_list:
                                add_phenotype_similarity(
                                    i,
                                    neighbors_column=f"neighbours_{knn}_knn",
                                    output_column=f"phenotype_similarity_ratio_{knn}_knn",
                                    phenotype_column=phenotype_column,
                                )

                    if self.calculate_phenotype_similarity_touching:
                        if (
                            not self.use_touching_neighbours_3d
                            or self.touching_dilation_um is None
                        ):
                            print(
                                "Skipping 3D touching phenotype similarity score: touching neighbours are disabled."
                            )
                        else:
                            touching_token = _format_param_token(
                                self.touching_dilation_um
                            )
                            touching_prefix = f"touching_neighbour_{touching_token}um"
                            print(
                                "Calculating phenotype similarity ratio based on 3D touching neighbours..."
                            )
                            add_phenotype_similarity(
                                i,
                                neighbors_column=f"{touching_prefix}_neighbours",
                                output_column=f"phenotype_similarity_ratio_{touching_prefix}",
                                phenotype_column=phenotype_column,
                            )

                    if self.do_phenotype_calling and self.create_split_phenotype_mask:
                        print("Creating split phenotype mask...")
                        split_phenotype_mask(
                            i,
                            self.phenotype_1,
                            self.phenotype_2,
                            self.phenotype_1_region,
                            self.phenotype_2_region,
                        )
                    print(
                        f"Finished processing sample {i}\n{idx + 1}/{len(self.sample_path_list)}"
                    )
                except Exception as sample_error:
                    import traceback

                    traceback.print_exc()
                    failed_samples.append(f"{i}: {sample_error}")
                    print(f"Skipping sample due to error: {i}")
                    formatted = traceback.format_exc().strip().splitlines()
                    last_line = (
                        formatted[-1].strip()
                        if formatted
                        else f"{type(sample_error).__name__}: {sample_error}"
                    )
                    if isinstance(sample_error, KeyError) and not _is_key_error(
                        last_line
                    ):
                        # Multi-line KeyError messages: make sure the UI can
                        # still recognise the type from this single line.
                        last_line = f"KeyError: {sample_error}".replace("\n", " ")
                    if _is_key_error(last_line):
                        print(CHANNEL_NAME_HINT)
                    self.sample_failed.emit(i, last_line)
                finally:
                    progress = int((global_idx + 1) / self.total_count * 100)
                    self.progress_updated.emit(progress)

            if failed_samples:
                print("Some samples failed during processing:")
                for failed in failed_samples:
                    print(f" - {failed}")

            elapsed_time = (datetime.now() - start_time).total_seconds()
            if self._stop_requested:
                self.stopped.emit(elapsed_time, stopped_at_idx)
            else:
                self.finished.emit(elapsed_time)
        except Exception as e:
            import traceback

            traceback.print_exc()  # Print full traceback
            self.error_occurred.emit(str(e))


def get_windows_theme():
    """Return True if the OS is in dark mode, False for light mode."""
    if sys.platform != "win32":
        return True  # default to dark on Linux
    try:
        registry_path = r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"
        registry_key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, registry_path)
        value, regtype = winreg.QueryValueEx(registry_key, "AppsUseLightTheme")
        winreg.CloseKey(registry_key)
        return value == 0  # 0 = dark mode, 1 = light mode
    except Exception as e:
        print(f"Could not detect Windows theme: {e}")
        return False


def get_additional_qss(size=12):
    """Generate stylesheet with customizable font size"""
    additional_qss = f"""
        QLabel {{
            font-size: {size}pt;
        }}
        QPushButton {{
            font-size: {size}pt;
        }}
        QLineEdit {{
            font-size: {size}pt;
        }}
        QComboBox {{
            font-size: {size}pt;
        }}
        QCheckBox {{
            font-size: {size}pt;
        }}
        QListWidget {{
            font-size: {size}pt;
        }}
        QProgressBar {{
            font-size: {size}pt;
        }}
        QTabBar {{
            font-size: {size}pt;
        }}
    """
    return additional_qss


if __name__ == "__main__":
    app = QApplication([])

    # Set desired font size here
    additional_qss = get_additional_qss(size=10)

    if get_windows_theme():
        qdarktheme.setup_theme(additional_qss=additional_qss)
    else:
        qdarktheme.setup_theme(theme="light", additional_qss=additional_qss)

    window = MainWindow()
    window.show()
    app.exec()
