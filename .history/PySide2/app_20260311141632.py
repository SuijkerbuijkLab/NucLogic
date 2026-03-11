from datetime import datetime
import sys
import os
from pathlib import Path
from time import time
import winreg

import tifffile
import matplotlib

matplotlib.use("Qt5Agg")

parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
from help_functions.file_to_folder import file_to_folder

# Import PySide2 FIRST
from PySide2.QtCore import Qt, QThread, Signal
from PySide2.QtGui import QIcon, QFont, QColor, QBrush
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
    QScrollArea,
)
import qdarktheme


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NucLogic")
        self.resize(1920, 1080)
        self.sample_list = None
        self.view_data_sample_list = None  # Separate list for View Data tab
        self.export_data_sample_list = None  # Separate list for Export Data tab
        self.samples_data = []  # Data structure holding samples
        self.selected_model_path = None
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

    def _create_segment_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        layout.setSpacing(30)

        layout.addLayout(self._create_channel_settings_menu())
        layout.addLayout(self._create_cropping_settings_layout())
        layout.addLayout(self._create_phenotype_settings_layout())
        layout.addLayout(self._create_advanced_settings_layout())
        layout.addLayout(self._create_run_segmentation_layout())
        layout.addLayout(self._create_progress_bar())

        widget.setLayout(layout)
        return widget

    def _create_cropping_settings_layout(self):
        layout = QVBoxLayout()
        layout.setSpacing(0)

        # Crop samples before segmentation checkbox
        self.do_crop_sample = QCheckBox(
            "Crop samples before segmentation, this is done automatically for timelapses and manually for fixed samples"
        )
        self.do_crop_sample.setChecked(True)
        layout.addWidget(self.do_crop_sample)

        # Save crop as ims or tif file selector
        layout2 = QHBoxLayout()
        layout2.addWidget(QLabel("Save cropped files as:"))
        self.save_crop_as = QComboBox()
        self.save_crop_as.addItems([".ims", ".tif"])
        layout2.addStretch()  # Push everything to the left
        layout2.addWidget(self.save_crop_as)
        layout.addLayout(layout2)

        return layout

    def _create_phenotype_settings_layout(self):
        layout = QVBoxLayout()
        layout.setSpacing(0)
        phenotype_settings_btn = QPushButton(
            "Show phenotype/cell-type calling settings"
        )
        phenotype_settings_btn.setCheckable(True)
        phenotype_settings_btn.toggled.connect(self._toggle_phenotype_settings)
        phenotype_settings_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        layout.addWidget(phenotype_settings_btn)

        # Create phenotype settings widget
        self.phenotype_settings_widget = QWidget()
        self.phenotype_settings_widget.setLayout(self._create_phenotype_settings())
        self.phenotype_settings_widget.setVisible(False)
        layout.addWidget(self.phenotype_settings_widget)
        return layout

    def _toggle_phenotype_settings(self, checked):
        """Toggle visibility of phenotype settings"""
        self.phenotype_settings_widget.setVisible(checked)
        self.do_phenotype_calling_checkbox.setChecked(
            checked
        )  # Sync with main checkbox

    def _create_phenotype_settings(self):
        """Add UI elements for phenotype/cell-type calling settings here"""
        layout = QVBoxLayout()
        layout.setSpacing(10)

        self.do_phenotype_calling_checkbox = QCheckBox(
            "Perform phenotype/cell-type calling"
        )
        # self.do_phenotype_calling_checkbox.setChecked(False)
        layout.addWidget(self.do_phenotype_calling_checkbox)

        layout2 = QHBoxLayout()
        layout2.addWidget(QLabel("Calculating phenotype/cell-type based on"))
        self.phenotype_1 = QComboBox()
        self.phenotype_2 = QComboBox()
        self._update_phenotype_combos()
        layout2.addWidget(self.phenotype_1)
        layout2.addWidget(QLabel("VS"))
        layout2.addWidget(self.phenotype_2)
        layout2.addStretch()  # Push everything to the left
        layout.addLayout(layout2)

        layout3 = QHBoxLayout()
        layout3.addWidget(QLabel("Cutoff calculation method:"))
        self.calculate_cutoff = QComboBox()
        self.calculate_cutoff.addItems(
            [
                "Calculate cutoff automatically (Usefull for populations that are roughly 50/50)",
                "Use fixed cutoff of 0 (Phenotype attributed to brightest channel, usefull for populations with clear positive/negative)",
            ]
        )
        layout3.addWidget(self.calculate_cutoff)
        layout3.addStretch()  # Push everything to the left
        layout.addLayout(layout3)

        layout4 = QHBoxLayout()
        layout4.addWidget(QLabel("Base phenotype/cell-type on:"))
        self.raw_or_background_subtracted = QComboBox()
        self.raw_or_background_subtracted.addItems(
            ["Background subtracted mean intensity values", "Raw mean intensity values"]
        )
        layout4.addWidget(self.raw_or_background_subtracted)
        layout4.addStretch()  # Push everything to the left
        layout.addLayout(layout4)

        return layout

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
        adv_settings_btn = QPushButton("Show advanced segmentation settings")
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

    def _create_channel_settings_menu(self):
        """Add UI elements for channel settings here"""
        layout = QVBoxLayout()
        layout.setSpacing(10)

        # Use grid layout for aligned columns
        self.channel_grid = QGridLayout()
        self.channel_grid.setSpacing(10)

        # Header row
        for text, col in [("Channel", 0), ("Type", 1), ("Name", 2)]:
            lbl = QLabel(text)
            lbl.setStyleSheet("font-weight: bold;")
            self.channel_grid.addWidget(lbl, 0, col)

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

    def get_phenotype_calling_settings(self):
        """Get phenotype calling settings"""
        do_calling = self.do_phenotype_calling_checkbox.isChecked()
        channel_1 = self.phenotype_1.currentText()
        channel_2 = self.phenotype_2.currentText()
        cutoff_method = self.calculate_cutoff.currentText()
        raw_or_background_subtracted = self.raw_or_background_subtracted.currentText()
        return (
            do_calling,
            channel_1,
            channel_2,
            cutoff_method,
            raw_or_background_subtracted,
        )

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
        (
            do_phenotype_calling,
            phenotype_1,
            phenotype_2,
            cutoff_method,
            raw_or_background_subtracted,
        ) = self.get_phenotype_calling_settings()
        do_crop_sample = self.do_crop_sample.isChecked()
        save_crop_as = self.save_crop_as.currentText()
        self.worker = SegmentationWorker(
            sample_path_list,
            model_path,
            channel_names,
            channel_types,
            breaking_threshold,
            do_phenotype_calling,
            phenotype_1,
            phenotype_2,
            cutoff_method,
            raw_or_background_subtracted,
            do_crop_sample,
            save_crop_as,
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
        )
        self.viewer_worker.data_loaded.connect(self._launch_napari_viewer)
        self.viewer_worker.error_occurred.connect(self._viewer_error)
        self.viewer_worker.start()

    def _launch_napari_viewer(self, data):
        import napari

        self.view_button.setEnabled(True)
        self.view_status_label.setVisible(False)

        channel_names = data["channel_names"]
        channel_colors = data["channel_colors"]
        show_segmentation = data["show_segmentation"]

        viewer = napari.Viewer(ndisplay=3)
        for sample_data in data["samples_data"]:
            sample = sample_data["sample"]
            movie = sample_data["movie"]
            voxel_size = sample_data["voxel_size"]
            segmentation = sample_data["segmentation"]

            for i, channel_name in enumerate(channel_names):
                viewer.add_image(
                    movie[:, i],
                    name=channel_name,
                    colormap=channel_colors[i].lower(),
                    visible=True,
                    blending="additive",
                    scale=voxel_size,
                )
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
                    )

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
        export_button = QPushButton("Export selected samples to TSV")
        export_button.clicked.connect(self.create_export_data)
        self.export_data_layout.addWidget(export_button)

        self.show_plots_button = QPushButton("Show Plots")
        self.show_plots_button.clicked.connect(self.show_plots)
        self.export_data_layout.addWidget(self.show_plots_button)

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

    def _load_full_results(self):
        import pandas as pd

        list_widget = self._get_list_widget(self.export_data_sample_list)
        selected_samples = [item.text() for item in list_widget.selectedItems()]
        full_results = []
        for sample in selected_samples:
            if not "properties" in os.listdir(
                os.path.join(self.path_text.text(), sample)
            ):
                continue
            properties = pd.read_csv(
                os.path.join(self.path_text.text(), sample, f"{sample}_properties.tsv"),
                sep="\t",
            )
            full_results.append(properties)
        return (
            pd.concat(full_results, ignore_index=True)
            if full_results
            else pd.DataFrame()
        )

    def create_export_data(self):
        full_results = self._load_full_results()
        if full_results.empty:
            return
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Exported Data",
            "",
            "TSV Files (*.tsv);;All Files (*)",
        )
        if save_path:
            full_results.to_csv(save_path, sep="\t", index=False)

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
            canvas.setAttribute(Qt.WA_TranslucentBackground)
            canvas.setMinimumHeight(400)
            self.plot_container_layout.addWidget(canvas)
            plt.close(fig)

        # --- Add your plots below ---
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor("none")
        ax.set_facecolor("none")

        cell_counts = (
            full_results.groupby(["organoid", "frame"])
            .size()
            .reset_index(name="cell_count")
        )
        sns.pointplot(
            data=cell_counts,
            x="frame",
            y="cell_count",
            hue="organoid",
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
                properties_folder = os.path.join(self.base_path, sample, "properties")
                if os.path.exists(properties_folder):
                    properties_sorted = sorted(
                        [
                            f
                            for f in os.listdir(properties_folder)
                            if f.endswith((".tsv", ".csv"))
                        ],
                        key=lambda x: int("".join(filter(str.isdigit, x))),
                    )
                    for prop_file in properties_sorted:
                        prop_path = os.path.join(properties_folder, prop_file)
                        sep = "\t" if prop_file.endswith(".tsv") else ","
                        frames.append(pd.read_csv(prop_path, sep=sep))
            result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            self.data_loaded.emit(result)
        except Exception as e:
            self.error_occurred.emit(str(e))


class ViewerWorker(QThread):
    data_loaded = Signal(object)
    error_occurred = Signal(str)

    def __init__(
        self, selected, base_path, channel_names, channel_colors, show_segmentation
    ):
        super().__init__()
        self.selected = selected
        self.base_path = base_path
        self.channel_names = channel_names
        self.channel_colors = channel_colors
        self.show_segmentation = show_segmentation

    def run(self):
        try:
            import numpy as np
            from utils import pad_to_shape
            from imaris_ims_file_reader.ims import ims

            samples_data = []
            for sample in self.selected:

                def _file_priority(f):
                    if f.endswith("_cropped.ims"):
                        return 0
                    if f.endswith("_cropped.tif"):
                        return 1
                    if f.endswith(".ims"):
                        return 2
                    if f.endswith(".tif"):
                        return 3  # plain .tif

                input_files = sorted(
                    [
                        f
                        for f in os.listdir(os.path.join(self.base_path, sample))
                        if f.endswith(("_cropped.ims", "_cropped.tif", ".ims", ".tif"))
                    ],
                    key=_file_priority,
                )

                input_file = input_files[0] if input_files else None

                print(input_file)

                if input_file and input_file.endswith(".ims"):
                    movie = ims(os.path.join(self.base_path, sample, input_file))
                    voxel_size = movie.resolution
                else:
                    with tifffile.TiffFile(input_file) as tif:
                        if tif.is_ome:
                            import xml.etree.ElementTree as ET

                            root = ET.fromstring(tif.ome_metadata)
                            ns = root.tag.split("}")[0].lstrip("{")
                            pixels = root.find(f".//{{{ns}}}Pixels")
                            voxel_size = (
                                float(pixels.get("PhysicalSizeZ", 1.0)),
                                float(pixels.get("PhysicalSizeY", 1.0)),
                                float(pixels.get("PhysicalSizeX", 1.0)),
                            )
                        else:
                            voxel_size = (1.0, 1.0, 1.0)
                        movie = tif.asarray()

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

                samples_data.append(
                    {
                        "sample": sample,
                        "movie": movie,
                        "voxel_size": voxel_size,
                        "segmentation": segmentation,
                    }
                )

            self.data_loaded.emit(
                {
                    "samples_data": samples_data,
                    "channel_names": self.channel_names,
                    "channel_colors": self.channel_colors,
                    "show_segmentation": self.show_segmentation,
                }
            )
        except Exception as e:
            import traceback

            traceback.print_exc()
            self.error_occurred.emit(str(e))


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
        do_phenotype_calling,
        phenotype_1,
        phenotype_2,
        cutoff_method,
        raw_or_background_subtracted,
        do_crop_sample,
        save_crop_as,
    ):
        super().__init__()
        self.sample_path_list = sample_path_list
        self.cell_model_path = cell_model_path
        self.channel_names = channel_names
        self.channel_types = channel_types
        self.breaking_threshold = breaking_threshold
        self.do_phenotype_calling = do_phenotype_calling
        self.phenotype_1 = phenotype_1
        self.phenotype_2 = phenotype_2
        self.cutoff_method = cutoff_method
        self.raw_or_background_subtracted = raw_or_background_subtracted
        self.do_crop_sample = do_crop_sample
        self.save_crop_as = save_crop_as

    def run(self):
        try:
            from main_functions.segment_organoid import segment_organoid
            from utils.load_model import load_model
            from main_functions.calculate_phenotypes import calculate_phenotypes
            from main_functions.crop_sample import crop_sample

            start_time = datetime.now()
            loaded_cell_model = load_model(self.cell_model_path)

            if self.do_crop_sample:
                from sam2.build_sam import build_sam2_video_predictor

                model_path = os.path.join(
                    os.path.dirname(os.path.dirname(__file__)), "models"
                )
                organoid_model = build_sam2_video_predictor(
                    os.path.join(model_path, "sam2.1_hiera_s.yaml"),
                    os.path.join(model_path, "sam2.1_hiera_small.pt"),
                )

            for idx, i in enumerate(self.sample_path_list):
                if self.do_crop_sample:
                    crop_sample(
                        i, self.channel_types, organoid_model, save_as=self.save_crop_as
                    )
                segment_organoid(
                    i,
                    loaded_cell_model,
                    self.channel_types,
                    self.channel_names,
                    self.breaking_threshold,
                    self.do_crop_sample,
                )
                print(self.do_phenotype_calling)
                if self.do_phenotype_calling:
                    print("Calculating phenotypes...")
                    calculate_phenotypes(
                        i,
                        self.phenotype_1,
                        self.phenotype_2,
                        self.cutoff_method,
                        self.raw_or_background_subtracted,
                    )
                print(
                    f"Finished processing sample {i}\n{idx + 1}/{len(self.sample_path_list)}"
                )
                progress = int((idx + 1) / len(self.sample_path_list) * 100)
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
