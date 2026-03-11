from datetime import datetime
import sys
import os
from pathlib import Path
from time import time
import winreg

parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
from help_functions.file_to_folder import file_to_folder

# Import PySide2 FIRST
from PySide2.QtCore import Qt, QThread, Signal
from PySide2.QtGui import QIcon, QFont
from PySide2.QtWidgets import (
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
    QAbstractItemView,
    QComboBox,
    QLineEdit,
    QGridLayout,
    QProgressBar,
    QCheckBox,
    QSizePolicy,
)
import qdarktheme


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nuclei Segmenter")
        self.resize(960, 540)
        self.sample_list = None
        self.view_data_sample_list = None  # Separate list for View Data tab
        self.samples_data = []  # Data structure holding samples
        self._set_icon()
        self._setup_ui()

    def _set_icon(self):
        icon_path = Path(__file__).parent / "www" / "organoid_segmenter.ico"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

    def _setup_ui(self):
        tabs = QTabWidget()
        tabs.addTab(self._create_load_data_tab(), "  Load Data  ")
        tabs.addTab(self._create_segment_tab(), "  Segment  ")

        # Create view data tab
        self.view_data_widget = QWidget()
        self.view_data_layout = QVBoxLayout()
        self.view_data_layout.setAlignment(Qt.AlignTop)
        self.view_data_layout.setSpacing(30)
        self.view_data_widget.setLayout(self.view_data_layout)
        tabs.addTab(self.view_data_widget, "  View Data  ")

        tabs.addTab(self._create_export_data_tab(), "  Export Data  ")
        tabs.setStyleSheet("QTabBar::tab { padding: 10px 20px; }")
        self.setCentralWidget(tabs)

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

    def browse_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder", "")
        if folder_path:
            self.path_text.setText(folder_path)
            self.update_file_info(folder_path)

    def update_file_info(self, dir_name):
        self._clear_layout(self.load_data_layout_2)
        self._clear_layout(self.load_data_layout_3)
        self._clear_layout(self.view_data_layout)

        if not Path(dir_name).exists():
            return

        # Store samples in data structure
        self.samples_data = self._get_samples(dir_name)
        image_files = self._get_image_files(dir_name)

        # Display samples count
        self.load_data_layout_2.addWidget(
            QLabel(f"Found {len(self.samples_data)} samples")
        )

        # Handle loose image files
        if image_files:
            self._add_loose_files_section(dir_name, image_files)

        # Create list widgets (only first time)
        if self.sample_list is None:
            self._create_sample_list_widgets()

        # Populate both lists with current samples data
        self._update_sample_lists()

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
        sample_list.setSelectionMode(QAbstractItemView.MultiSelection)
        sample_list.itemSelectionChanged.connect(self._update_segment_label)
        layout.addWidget(sample_list)

        widget.setLayout(layout)
        return widget

    def _get_samples(self, dir_name):
        """Get list of subdirectories (samples)"""
        return [
            d for d in os.listdir(dir_name) if os.path.isdir(os.path.join(dir_name, d))
        ]

    def _get_image_files(self, dir_name):
        """Get list of image files in root directory"""
        return [
            f for f in os.listdir(dir_name) if f.endswith((".tif", ".tiff", ".ims"))
        ]

    def _add_loose_files_section(self, dir_name, image_files):
        """Add section for handling loose image files"""
        self.load_data_layout_2_1.addWidget(
            QLabel(
                f"Found {len(image_files)} image files without a directory. "
                "Create directories for them?"
            )
        )
        create_button = QPushButton("Create directories")
        create_button.clicked.connect(lambda: self._handle_create_dirs(dir_name))
        self.load_data_layout_2_1.addWidget(create_button)
        self.load_data_layout_2.addLayout(self.load_data_layout_2_1)

    def _handle_create_dirs(self, dir_name):
        file_to_folder(dir_name)
        self.update_file_info(dir_name)

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
        """Keep selections in sync between both lists"""
        load_data_list = self._get_list_widget(self.sample_list)
        view_data_list = self._get_list_widget(self.view_data_sample_list)

        # Clear view_data selections and set them to match load_data
        view_data_list.blockSignals(True)
        view_data_list.clearSelection()
        for item in load_data_list.selectedItems():
            matching_items = view_data_list.findItems(item.text(), Qt.MatchExactly)
            if matching_items:
                matching_items[0].setSelected(True)
        view_data_list.blockSignals(False)

    def select_all_samples(self):
        self.sample_list.findChild(QListWidget).selectAll()
        self._sync_selections()

    def deselect_all_samples(self):
        self.sample_list.findChild(QListWidget).clearSelection()
        self._sync_selections()

    def get_selected_samples(self):
        """Get list of selected sample names"""
        list_widget = self._get_list_widget(self.sample_list)
        return [item.text() for item in list_widget.selectedItems()]

    def _create_segment_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        layout.setSpacing(30)

        layout.addLayout(self._create_channel_settings_menu())
        layout.addLayout(self._create_advanced_settings_layout())
        layout.addLayout(self._create_run_segmentation_layout())
        layout.addLayout(self._create_progress_bar())

        widget.setLayout(layout)
        return widget

    def _create_run_segmentation_layout(self):
        layout = QVBoxLayout()
        layout.setSpacing(5)
        self.segment_tab_label = QLabel("Selected 0 samples for segmentation")
        layout.addWidget(self.segment_tab_label)
        self.run_segmentation_btn = QPushButton("Run segmentation")
        self.run_segmentation_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.run_segmentation_btn.clicked.connect(self.start_segmentation)
        layout.addWidget(self.run_segmentation_btn)
        return layout

    def _update_segment_label(self):
        """Update the segment tab label with selected samples count"""
        count = len(self.get_selected_samples())
        self.segment_tab_label.setText(f"Selected {count} samples for segmentation")

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
        return layout

    def _create_advanced_settings_layout(self):
        layout = QVBoxLayout()
        layout.setSpacing(0)
        adv_settings_btn = QPushButton("Show advanced settings")
        adv_settings_btn.setCheckable(True)
        adv_settings_btn.toggled.connect(self._toggle_advanced_settings)
        adv_settings_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        layout.addWidget(adv_settings_btn)

        # Create advanced settings widget
        self.advanced_settings_widget = QWidget()
        self.advanced_settings_widget.setLayout(self._create_advanced_settings())
        self.advanced_settings_widget.setVisible(False)
        layout.addWidget(self.advanced_settings_widget)
        return layout

    def _toggle_advanced_settings(self, checked):
        """Toggle visibility of advanced settings"""
        self.advanced_settings_widget.setVisible(checked)

    def _create_channel_settings_menu(self):
        """Add UI elements for channel settings here"""
        layout = QVBoxLayout()
        layout.setSpacing(10)

        # Use grid layout for aligned columns
        self.channel_grid = QGridLayout()
        self.channel_grid.setSpacing(10)

        # Header row
        self.channel_grid.addWidget(QLabel("Channel"), 0, 0)
        self.channel_grid.addWidget(QLabel("Type"), 0, 1)
        self.channel_grid.addWidget(QLabel("Name"), 0, 2)

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
        self.channel_grid.addWidget(line_edit, row, 2)

        # Store widgets for later deletion
        self.channel_widgets[channel_number] = (label, combo_box, line_edit)

    def add_channel(self):
        """Add a new channel settings line"""
        self.channel_count += 1
        self._add_channel_grid_row(self.channel_count)

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

    def _create_advanced_settings(self):
        """Add UI elements for advanced settings here"""
        self.advanced_layout = QVBoxLayout()
        self.advanced_layout.setSpacing(10)
        self.advanced_layout.addLayout(self._create_model_settings())
        self.advanced_layout.addLayout(self._create_save_individual_files())
        breaking_threshold_layout = QHBoxLayout()
        breaking_threshold_layout.addWidget(
            QLabel("Breaking threshold for cell stitching:")
        )
        self.breaking_threshold_input = QLineEdit("2.5")
        breaking_threshold_layout.addWidget(self.breaking_threshold_input)
        self.advanced_layout.addLayout(breaking_threshold_layout)
        return self.advanced_layout

    def _create_model_settings(self):
        model_setting_layout = QHBoxLayout()
        model_setting_layout.addWidget(QLabel("2D Segmentation model:"))
        self.model_combo_box = QComboBox()  # Store as instance variable
        self.model_combo_box.addItems(
            ["High quality imaging", "Low quality imaging", "Custom model"]
        )
        self.model_combo_box.currentTextChanged.connect(
            self._on_model_selection_changed
        )
        model_setting_layout.addWidget(self.model_combo_box)
        return model_setting_layout

    def _on_model_selection_changed(self, selected_model):
        """Handle model selection change"""
        # Remove existing custom model section if present
        if (
            hasattr(self, "custom_model_layout")
            and self.custom_model_layout is not None
        ):
            self._clear_layout(self.custom_model_layout)
            self.advanced_layout.removeItem(self.custom_model_layout)
            self.custom_model_layout = None

        # Reset custom model path when switching models
        self.selected_model_path = None

        # Add custom model section if selected
        if selected_model == "Custom model":
            self.custom_model_layout = self._create_custom_model_selection()
            self.advanced_layout.insertLayout(1, self.custom_model_layout)

    def _create_custom_model_selection(self):
        """Create UI for selecting a custom model file"""
        layout = QHBoxLayout()
        layout.addWidget(QLabel("Select custom model file:"))

        self.model_file_label = QLineEdit()
        self.model_file_label.setReadOnly(True)
        self.model_file_label.setPlaceholderText("No file selected")
        layout.addWidget(self.model_file_label)

        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self._browse_model_file)
        layout.addWidget(browse_button)

        return layout

    def _browse_model_file(self):
        """Open file dialog to select model file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model File",
            "",
            "Model Files (*.pt *.pth *.onnx);;All Files (*)",
        )
        if file_path:
            self.model_file_label.setText(file_path)
            self.selected_model_path = file_path

    def _create_save_individual_files(self):
        layout = QVBoxLayout()
        frames = QCheckBox("Save individual frames")
        frames.setChecked(True)
        layout.addWidget(frames)
        segmentation = QCheckBox("Save individual segmentation masks")
        segmentation.setChecked(True)
        layout.addWidget(segmentation)
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
        """Get the model path based on selected model"""
        selected_model = self.model_combo_box.currentText()

        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

        if selected_model == "High quality imaging":
            return os.path.join(model_path, "2d_high_quality_model")
        elif selected_model == "Low quality imaging":
            return os.path.join(model_path, "2d_low_quality_model")
        elif selected_model == "Custom model":
            if hasattr(self, "selected_model_path") and self.selected_model_path:
                return self.selected_model_path
            else:
                return None  # No custom model selected

    def get_sample_path_list(self):
        """Get list of selected samples"""
        samples = self.get_selected_samples()
        base_path = self.path_text.text()
        return [os.path.join(base_path, sample) for sample in samples]

    def get_breaking_threshold(self):
        try:
            return float(self.breaking_threshold_input.text())
        except ValueError:
            return 2.5  # Default value if input is invalid

    def start_segmentation(self):
        """Start segmentation in a separate thread"""
        self.progressbar.setVisible(True)
        self.progressbar.setValue(0)
        self.run_segmentation_btn.setEnabled(False)
        self.segmentation_summary_label.setVisible(False)

        sample_path_list = self.get_sample_path_list()
        model_path = self.get_model_path()
        channel_names, channel_types = self.get_channel_settings()
        breaking_threshold = self.get_breaking_threshold()
        self.worker = SegmentationWorker(
            sample_path_list,
            model_path,
            channel_names,
            channel_types,
            breaking_threshold,
        )
        self.worker.progress_updated.connect(self.progressbar.setValue)
        self.worker.finished.connect(self.segmentation_finished)
        self.worker.error_occurred.connect(
            self.segmentation_error
        )  # Connect error signal
        self.worker.start()

    def segmentation_finished(self, elapsed_time):
        """Called when segmentation finishes"""
        self.run_segmentation_btn.setEnabled(True)

        # Format the summary
        finish_time = datetime.now().strftime("%H:%M:%S")
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)

        summary_text = f"Segmented {len(self.get_selected_samples())} samples.\nSegmentation completed at {finish_time} (took {minutes}m {seconds}s)"
        self.segmentation_summary_label.setText(summary_text)
        self.segmentation_summary_label.setVisible(True)

    def segmentation_error(self, error_message):
        """Called when an error occurs during segmentation"""
        self.run_segmentation_btn.setEnabled(True)
        self.progressbar.setVisible(False)

        self.segmentation_summary_label.setText(f"Error: {error_message}")
        self.segmentation_summary_label.setVisible(True)

    def _setup_view_data_widgets(self):
        """Setup all widgets for View Data tab"""
        # Sample selection section
        sample_section = QVBoxLayout()
        sample_section.setSpacing(5)
        sample_section.addWidget(QLabel("Select samples:"))
        self.view_data_sample_list = self._create_single_sample_list()
        sample_section.addWidget(self.view_data_sample_list)
        self.view_data_layout.addLayout(sample_section)

        # Analysis Options section
        analysis_section = QVBoxLayout()
        analysis_section.setSpacing(5)
        analysis_section.addWidget(QLabel("Analysis Options:"))

        # Channel selector
        channel_layout = QHBoxLayout()
        channel_layout.setSpacing(10)
        channel_layout.addWidget(QLabel("Channel:"))
        self.channel_selector = QComboBox()
        self.channel_selector.addItems(["Channel 1", "Channel 2", "Channel 3"])
        channel_layout.addWidget(self.channel_selector)
        analysis_section.addLayout(channel_layout)

        self.view_data_layout.addLayout(analysis_section)

        # Buttons section
        button_section = QVBoxLayout()
        button_section.setSpacing(5)
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        view_button = QPushButton("View Selected")
        export_button = QPushButton("Export Images")
        view_button.clicked.connect(self.view_selected_samples)
        export_button.clicked.connect(self.export_images)
        button_layout.addWidget(view_button)
        button_layout.addWidget(export_button)
        button_section.addLayout(button_layout)
        self.view_data_layout.addLayout(button_section)

        self.view_data_layout.addStretch()

    def view_selected_samples(self):
        """Handle view button"""
        selected = self.get_selected_samples()
        overlay = self.overlay_checkbox.isChecked()
        channel = self.channel_selector.currentText()
        print(f"Viewing {selected} with {channel}, overlay={overlay}")

    def export_images(self):
        """Handle export button"""
        selected = self.get_selected_samples()
        print(f"Exporting {selected}")

    def _create_export_data_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        layout.setSpacing(30)

        export_button = QPushButton("Export all data to CSV")
        # export_button.clicked.connect(self.export_data)
        layout.addWidget(export_button)

        widget.setLayout(layout)
        return widget


class SegmentationWorker(QThread):
    progress_updated = Signal(int)
    finished = Signal(float)
    error_occurred = Signal(str)  # Add error signal

    def __init__(
        self,
        sample_path_list,
        cell_model_path,
        channel_names,
        channel_types,
        breaking_threshold,
    ):
        super().__init__()
        self.sample_path_list = sample_path_list
        self.cell_model_path = cell_model_path
        self.channel_names = channel_names
        self.channel_types = channel_types
        self.breaking_threshold = breaking_threshold

    def run(self):
        try:
            from main_functions.segment_organoid import segment_organoid
            from utils.load_model import load_model

            start_time = datetime.now()
            loaded_cell_model = load_model(self.cell_model_path)
            for i in self.sample_path_list:
                segment_organoid(
                    i,
                    loaded_cell_model,
                    self.channel_types,
                    self.channel_names,
                    self.breaking_threshold,
                )
                progress = int(
                    ((self.sample_path_list.index(i) + 1) / len(self.sample_path_list))
                    * 100
                )
                self.progress_updated.emit(progress)

            elapsed_time = (datetime.now() - start_time).total_seconds()
            self.finished.emit(elapsed_time)
        except Exception as e:
            import traceback

            traceback.print_exc()  # Print full traceback
            self.error_occurred.emit(str(e))


def get_windows_theme():
    """Detect if Windows is in dark mode"""
    try:
        registry_path = r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"
        registry_key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, registry_path)
        value, regtype = winreg.QueryValueEx(registry_key, "AppsUseLightTheme")
        winreg.CloseKey(registry_key)
        return value == 0  # 0 = dark mode, 1 = light mode
    except Exception as e:
        print(f"Could not detect Windows theme: {e}")
        return False  # Default to light mode


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
    app.exec_()
