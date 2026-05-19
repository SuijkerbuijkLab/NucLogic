from datetime import datetime
import sys
import os
import shutil
import torch
from pathlib import Path
import winreg

import tifffile
import matplotlib

matplotlib.use("Qt5Agg")

parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
from utils.file_to_folder import file_to_folder
from utils.tiff_metadata import load_tiff_movie_and_metadata

# Import PySide2 FIRST
from PySide2.QtCore import Qt, QThread, QTimer, Signal
from PySide2.QtGui import QIcon, QColor, QBrush
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
    QListWidgetItem,
    QAbstractItemView,
    QComboBox,
    QLineEdit,
    QGridLayout,
    QProgressBar,
    QCheckBox,
    QSizePolicy,
    QScrollArea,
    QFrame,
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

    def _make_separator(self):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        return line

    def _info_label(self, tooltip_text):
        lbl = QLabel("ⓘ")
        lbl.setToolTip(tooltip_text)
        lbl.setStyleSheet("color: #5599CC; font-size: 20px;")
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
        self.adv_stats_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        layout.addWidget(self.adv_stats_btn)

        # Create advanced statistics settings widget
        self.advanced_statistics_widget = QWidget()
        self.advanced_statistics_widget.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Maximum
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
        self.nuclei_or_cytoplasm_checkbox = QComboBox()
        self.nuclei_or_cytoplasm_checkbox.addItems(
            ["Nuclei", "Cytoplasm", "Whole cell"]
        )
        self.nuclei_or_cytoplasm_checkbox.currentTextChanged.connect(
            self._cytoplasm_measurement_toggled
        )
        layout1.addWidget(self.nuclei_or_cytoplasm_checkbox)
        layout1.addWidget(self._info_label(
            "Phenotype/cell-type calling uses this same nuclei/cytoplasm/whole-cell intensity setting."
        ))
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
            QAbstractItemView.MultiSelection
        )
        self.calculate_advanced_statistics_list.setMaximumHeight(180)

        advanced_property_items = [
            "area_bbox",
            "area_convex",
            "area_filled",
            "axis_major_length",
            "axis_minor_length",
            "centroid_local",
            "centroid_weighted",
            "centroid_weighted_local",
            "coords_scaled",
            "coords",
            "equivalent_diameter_area",
            "euler_number",
            "extent",
            "feret_diameter_max",
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
            "K-Nearest Neighbours (KNN): finds the K cells whose centroids are closest in 3D space and treats them as neighbours. "
            "This is a purely distance-based approach — cells do not need to be physically touching."
        ))
        knn_row.addStretch()
        neighbour_sub_layout.addWidget(self.knn_checkbox_row_widget)

        self.knn_neighbour_options_widget = QWidget()
        knn_neighbour_options_layout = QVBoxLayout(self.knn_neighbour_options_widget)
        knn_neighbour_options_layout.setContentsMargins(30, 0, 0, 0)
        knn_neighbour_options_layout.setSpacing(6)

        layout_knn = QHBoxLayout()
        layout_knn.addWidget(QLabel("Number of nearest neighbours (K):"))
        self.knn_input = QLineEdit()
        self.knn_input.setPlaceholderText("e.g. 5")
        self.knn_input.setMaximumWidth(120)
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
            "then checks which other nuclei overlap with the expanded region. "
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
            "Calculates what fraction of a cell's touching neighbours share the same phenotype/cell-type. "
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

    def _cytoplasm_measurement_toggled(self):
        selected_option = self.nuclei_or_cytoplasm_checkbox.currentText()
        if selected_option == "Cytoplasm" or selected_option == "Whole cell":
            self.QLabel_cytoplasm_size.setVisible(True)
            self.cytoplasm_size_input.setVisible(True)
            self.save_measurement_mask_checkbox.setVisible(True)
        else:
            self.QLabel_cytoplasm_size.setVisible(False)
            self.cytoplasm_size_input.setVisible(False)
            self.save_measurement_mask_checkbox.setVisible(False)
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
        measure_intensity_in = self.nuclei_or_cytoplasm_checkbox.currentText()
        cytoplasm_size = 5
        if measure_intensity_in in ("Cytoplasm", "Whole cell"):
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
        knn = None
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
                        "KNN is enabled. Please enter a positive integer K value."
                    )
                try:
                    knn = int(knn_text)
                except ValueError as exc:
                    raise ValueError(
                        "Invalid KNN value. Please enter a positive integer."
                    ) from exc
                if knn < 1:
                    raise ValueError(
                        "Invalid KNN value. Please enter a positive integer."
                    )
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

        voxel_z_text = self.voxel_size_z_input.text().strip()
        voxel_x_text = self.voxel_size_x_input.text().strip()
        voxel_y_text = self.voxel_size_y_input.text().strip()
        if voxel_z_text or voxel_x_text or voxel_y_text:
            try:
                voxel_z = float(voxel_z_text) if voxel_z_text else 1.0
                voxel_x = float(voxel_x_text) if voxel_x_text else 1.0
                voxel_y = float(voxel_y_text) if voxel_y_text else 1.0
            except ValueError as exc:
                raise ValueError(
                    "Invalid voxel size override. Please enter valid float values for Z, X, and Y."
                ) from exc
            if voxel_z <= 0 or voxel_x <= 0 or voxel_y <= 0:
                raise ValueError(
                    "Invalid voxel size override. Z, X, and Y must be positive values."
                )
            user_voxel_size = (voxel_z, voxel_y, voxel_x)
        else:
            user_voxel_size = (1.0, 1.0, 1.0)
        return (
            extra_props,
            advanced_statistics_only,
            measure_intensity_in,
            cytoplasm_size,
            save_measurement_mask,
            user_voxel_size,
            calculate_neighbour_statistics,
            use_knn_neighbours,
            knn,
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
        self.crop_config_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        layout.addWidget(self.crop_config_btn)

        self.crop_config_widget = QWidget()
        self.crop_config_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
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

        # Crop samples before segmentation checkbox
        self.do_crop_sample = QCheckBox("Crop samples before segmentation")
        self.do_crop_sample.setChecked(True)
        self.do_crop_sample.toggled.connect(self._toggle_crop_suboptions)
        crop_row = QHBoxLayout()
        crop_row.addWidget(self.do_crop_sample)
        crop_row.addWidget(self._info_label(
            "Timelapses are cropped automatically; fixed samples can use manual or automatic cropping."
        ))
        crop_row.addStretch()
        layout.addLayout(crop_row)

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

        self.skip_crop_existing_files_checkbox = QCheckBox(
            "Skip cropping — cropped files already exist"
        )
        self.skip_crop_existing_files_checkbox.setChecked(False)
        sub_layout.addWidget(self.skip_crop_existing_files_checkbox)

        combo_row = QHBoxLayout()
        combo_row.setContentsMargins(0, 2, 0, 0)
        combo_row.addWidget(QLabel("Save cropped files as:"))
        self.save_crop_as = QComboBox()
        self.save_crop_as.addItems([".ims", ".tif"])
        combo_row.addWidget(self.save_crop_as)
        combo_row.addWidget(self._info_label(
            "Use .ims if you work with Imaris; use .tif for universal compatibility."
        ))
        combo_row.addStretch()
        sub_layout.addLayout(combo_row)

        layout.addWidget(self.crop_suboptions_widget)
        self._toggle_crop_suboptions(self.do_crop_sample.isChecked())

        return layout

    def _toggle_crop_suboptions(self, checked):
        self.crop_suboptions_widget.setVisible(checked)

    def _create_phenotype_settings_layout(self):
        layout = QVBoxLayout()
        layout.setSpacing(0)
        self.phenotype_settings_btn = QPushButton(
            "Show phenotype/cell-type calling settings"
        )
        self.phenotype_settings_btn.setCheckable(True)
        self.phenotype_settings_btn.toggled.connect(self._toggle_phenotype_settings)
        self.phenotype_settings_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
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
        layout2.addWidget(self.phenotype_1)
        layout2.addWidget(QLabel("VS"))
        layout2.addWidget(self.phenotype_2)
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
        self.run_segmentation_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.run_segmentation_btn.clicked.connect(lambda: self.start_segmentation())
        self.stop_segmentation_btn = QPushButton("Stop after current sample")
        self.stop_segmentation_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.stop_segmentation_btn.setVisible(False)
        self.stop_segmentation_btn.clicked.connect(self._request_stop)
        self.continue_segmentation_btn = QPushButton("Continue segmentation")
        self.continue_segmentation_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.continue_segmentation_btn.setVisible(False)
        self.continue_segmentation_btn.clicked.connect(self._continue_segmentation)
        save_config_btn = QPushButton("Save configuration")
        save_config_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        save_config_btn.clicked.connect(self.save_configuration)
        load_config_btn = QPushButton("Load configuration")
        load_config_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
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
        self.adv_settings_btn = QPushButton("Show advanced segmentation settings")
        self.adv_settings_btn.setCheckable(True)
        self.adv_settings_btn.toggled.connect(self._toggle_advanced_settings)
        self.adv_settings_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
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
        self.breaking_threshold_input = QLineEdit("2.5")
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

        voxel_label_row = QHBoxLayout()
        voxel_label_row.addWidget(QLabel("Voxel size override (Z / X / Y in µm):"))
        voxel_label_row.addWidget(self._info_label(
            "Only needed if your image file has incorrect or missing spatial metadata.\n"
            "Leave blank to use the voxel size read from the file automatically."
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
        self.channel_config_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        btn_row.addWidget(self.channel_config_btn)
        btn_row.addWidget(self._info_label(
            "Configure ALL channels present in your image/movie — not only the channel(s) you want to segment."
        ))
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.channel_config_widget = QWidget()
        self.channel_config_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
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
        model_setting_layout.addWidget(self._info_label(
            "Pretrained 2D Cellpose segmentation models used to detect nuclei in each Z-slice. "
            "The slices are then stitched into a full 3D segmentation. "
            "You can upload a custom Cellpose model trained on your own data."
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
            if self._is_sam_support_file(entry):
                continue
            full_path = os.path.join(models_dir, entry)
            if os.path.isdir(full_path) or os.path.isfile(full_path):
                model_entries.append((entry, full_path))

        self.model_combo_box.blockSignals(True)
        self.model_combo_box.clear()
        for display_name, full_path in model_entries:
            self.model_combo_box.addItem(display_name, full_path)

        if model_entries:
            if select_name:
                idx = self.model_combo_box.findText(select_name)
                self.model_combo_box.setCurrentIndex(idx if idx >= 0 else 0)
            else:
                self.model_combo_box.setCurrentIndex(0)
        self.model_combo_box.blockSignals(False)

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
        """Get the model path based on selected model"""
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
            return 2.5  # Default value if input is invalid

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
        QApplication.processEvents()

        model_path = self.get_model_path()
        if model_path is None:
            self.segmentation_error(
                "No segmentation model found. Please upload a model in the advanced segmentation settings."
            )
            return
        channel_names, channel_types = self.get_channel_settings()
        breaking_threshold = self.get_breaking_threshold()
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
            ) = self.get_phenotype_calling_settings()
        except ValueError as e:
            self.segmentation_error(str(e))
            return
        do_crop_sample = self.do_crop_sample.isChecked()
        manual_crop_fixed = self.manual_crop_fixed_checkbox.isChecked()
        skip_crop_existing_files = self.skip_crop_existing_files_checkbox.isChecked()
        save_crop_as = self.save_crop_as.currentText()
        save_frames = self.save_frames_checkbox.isChecked()
        save_segmentation = self.save_segmentation_checkbox.isChecked()
        try:
            (
                extra_props,
                advanced_statistics_only,
                measure_intensity_in,
                cytoplasm_size,
                save_measurement_mask,
                user_voxel_size,
                calculate_neighbour_statistics,
                use_knn_neighbours,
                knn,
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
            and not skip_crop_existing_files
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

                for sample_path in sample_path_list:
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
            skip_crop_existing_files,
            save_crop_as,
            save_frames,
            save_segmentation,
            extra_props,
            advanced_statistics_only,
            measure_intensity_in,
            cytoplasm_size,
            save_measurement_mask,
            user_voxel_size,
            calculate_neighbour_statistics,
            use_knn_neighbours,
            knn,
            calculate_phenotype_similarity_knn,
            use_touching_neighbours_3d,
            touching_dilation_um,
            calculate_phenotype_similarity_touching,
            manually_cropped_fixed_samples,
            index_offset=index_offset,
            total_count=total_count,
        )
        self.worker.progress_updated.connect(self.progressbar.setValue)
        self.worker.finished.connect(self.segmentation_finished)
        self.worker.stopped.connect(self.segmentation_stopped)
        self.worker.error_occurred.connect(self.segmentation_error)
        self.worker.heartbeat.connect(self._on_worker_heartbeat)
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

        self.segmentation_summary_label.setStyleSheet("color: #44BB44;")
        summary_text = f"Segmented {len(self.get_selected_samples())} samples.\nSegmentation completed at {finish_time} (took {minutes}m {seconds}s)"
        self.segmentation_summary_label.setText(summary_text)
        self.segmentation_summary_label.setVisible(True)

    def segmentation_stopped(self, elapsed_time, stopped_idx):
        """Called when the user stops segmentation between samples"""
        self._segmentation_running = False
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
        save_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        save_btn.clicked.connect(self.save_configuration)
        load_btn = QPushButton("Load configuration")
        load_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
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
            "do_crop_sample": self.do_crop_sample.isChecked(),
            "manual_crop_fixed": self.manual_crop_fixed_checkbox.isChecked(),
            "skip_crop_existing_files": self.skip_crop_existing_files_checkbox.isChecked(),
            "save_crop_as": self.save_crop_as.currentText(),
            "run_mode": self.run_mode_combo.currentText(),
            "breaking_threshold": self.breaking_threshold_input.text(),
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
            "measure_intensity_in": self.nuclei_or_cytoplasm_checkbox.currentText(),
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

        _set_check(self.do_crop_sample, "do_crop_sample", True)
        _set_check(self.manual_crop_fixed_checkbox, "manual_crop_fixed")
        _set_check(self.skip_crop_existing_files_checkbox, "skip_crop_existing_files")
        _set_combo(self.save_crop_as, "save_crop_as", ".ims")
        _set_combo(self.run_mode_combo, "run_mode", "Full pipeline")
        self.breaking_threshold_input.setText(config.get("breaking_threshold", "2.5"))
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
        _set_combo(self.nuclei_or_cytoplasm_checkbox, "measure_intensity_in", "Nuclei")
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
        self._cytoplasm_measurement_toggled()
        self._toggle_crop_suboptions(self.do_crop_sample.isChecked())
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
        self.export_button = QPushButton("Export selected samples to TSV")
        self.export_button.clicked.connect(self.create_export_data)
        self.export_data_layout.addWidget(self.export_button)

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

    def _export_error(self, error_message):
        self.export_button.setEnabled(True)
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
            canvas.setAttribute(Qt.WA_TranslucentBackground)
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

            frames = []
            for sample in self.selected_samples:
                prop_path = os.path.join(
                    self.base_path, sample, f"{sample}_properties.tsv"
                )
                if os.path.exists(prop_path):
                    frames.append(pd.read_csv(prop_path, sep="\t"))
            if frames:
                pd.concat(frames, ignore_index=True).to_csv(
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
                    movie, voxel_size, _, _ = load_tiff_movie_and_metadata(
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
    stopped = Signal(float, int)
    error_occurred = Signal(str)
    heartbeat = Signal(str)

    def __init__(
        self,
        sample_path_list,
        cell_model_path,
        channel_names,
        channel_types,
        breaking_threshold,
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
        skip_crop_existing_files,
        save_crop_as,
        save_frames,
        save_segmentation,
        extra_props,
        advanced_statistics_only,
        measure_intensity_in,
        cytoplasm_size,
        save_measurement_mask,
        user_voxel_size,
        calculate_neighbour_statistics=False,
        use_knn_neighbours=False,
        knn=None,
        calculate_phenotype_similarity_knn=False,
        use_touching_neighbours_3d=False,
        touching_dilation_um=None,
        calculate_phenotype_similarity_touching=False,
        manually_cropped_fixed_samples=None,
        index_offset=0,
        total_count=None,
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
        self.do_phenotype_calling = do_phenotype_calling
        self.phenotype_calling_only = phenotype_calling_only
        self.create_split_phenotype_mask = create_split_phenotype_mask
        self.phenotype_1 = phenotype_1
        self.phenotype_2 = phenotype_2
        self.cutoff_method = cutoff_method
        self.custom_cutoff = custom_cutoff
        self.raw_or_background_subtracted = raw_or_background_subtracted
        self.do_crop_sample = do_crop_sample
        self.manual_crop_fixed = manual_crop_fixed
        self.skip_crop_existing_files = skip_crop_existing_files
        self.save_crop_as = save_crop_as
        self.save_frames = save_frames
        self.save_segmentation = save_segmentation
        self.extra_props = extra_props
        self.advanced_statistics_only = advanced_statistics_only
        self.measure_intensity_in = measure_intensity_in
        self.cytoplasm_size = cytoplasm_size
        self.save_measurement_mask = save_measurement_mask
        self.user_voxel_size = user_voxel_size
        self.calculate_neighbour_statistics = calculate_neighbour_statistics
        self.use_knn_neighbours = use_knn_neighbours
        self.knn = knn
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

    def run(self):
        try:
            from main_functions.segment_organoid import segment_organoid
            from utils.load_model import load_model
            from main_functions.calculate_phenotypes import calculate_phenotypes
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
                self.do_crop_sample
                and not self.skip_crop_existing_files
                and not self.phenotype_calling_only
                and not self.advanced_statistics_only
                and any(
                    os.path.normcase(os.path.normpath(sample_path))
                    not in self.manually_cropped_fixed_samples
                    for sample_path in self.sample_path_list
                )
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
                        if self.do_crop_sample:
                            if self.skip_crop_existing_files:
                                print(
                                    f"Skipping crop for {i}: using existing cropped files."
                                )
                            else:
                                normalized_i = os.path.normcase(os.path.normpath(i))
                                if normalized_i in self.manually_cropped_fixed_samples:
                                    print(
                                        f"Skipping crop for {i}: fixed sample was manually cropped in UI thread."
                                    )
                                else:
                                    crop_sample(
                                        i,
                                        self.channel_types,
                                        organoid_model,
                                        manual_fixed=self.manual_crop_fixed,
                                        save_as=self.save_crop_as,
                                        user_voxel_size=self.user_voxel_size,
                                    )
                        segment_organoid(
                            i,
                            loaded_cell_model,
                            self.channel_types,
                            self.channel_names,
                            self.breaking_threshold,
                            self.do_crop_sample,
                            save_frames=self.save_frames,
                            save_segmentation=self.save_segmentation,
                            measure_intensity_in=self.measure_intensity_in,
                            cytoplasm_size=self.cytoplasm_size,
                            save_measurement_mask=self.save_measurement_mask,
                            user_voxel_size=self.user_voxel_size,
                        )

                        should_run_advanced_stats = bool(self.extra_props) or (
                            self.calculate_neighbour_statistics
                            and (
                                (self.use_knn_neighbours and self.knn is not None)
                                or (
                                    self.use_touching_neighbours_3d
                                    and self.touching_dilation_um is not None
                                )
                            )
                        )
                        if should_run_advanced_stats:
                            print("Calculating advanced statistics...")
                            add_advanced_statistics(
                                i,
                                self.extra_props,
                                self.channel_names,
                                do_crop_sample=self.do_crop_sample,
                                measure_intensity_in=self.measure_intensity_in,
                                cytoplasm_size=self.cytoplasm_size,
                                save_measurement_mask=self.save_measurement_mask,
                                user_voxel_size=self.user_voxel_size,
                                calculate_neighbour_statistics=self.calculate_neighbour_statistics,
                                use_knn_neighbours=self.use_knn_neighbours,
                                knn=self.knn,
                                use_touching_neighbours_3d=self.use_touching_neighbours_3d,
                                touching_dilation_um=self.touching_dilation_um,
                            )
                    if self.do_phenotype_calling:
                        print("Calculating phenotypes...")
                        calculate_phenotypes(
                            i,
                            self.phenotype_1,
                            self.phenotype_2,
                            self.cutoff_method,
                            self.custom_cutoff,
                            self.raw_or_background_subtracted,
                        )
                    if self.advanced_statistics_only:
                        print("Calculating advanced statistics...")
                        add_advanced_statistics(
                            i,
                            self.extra_props,
                            self.channel_names,
                            do_crop_sample=self.do_crop_sample,
                            measure_intensity_in=self.measure_intensity_in,
                            cytoplasm_size=self.cytoplasm_size,
                            save_measurement_mask=self.save_measurement_mask,
                            user_voxel_size=self.user_voxel_size,
                            calculate_neighbour_statistics=self.calculate_neighbour_statistics,
                            use_knn_neighbours=self.use_knn_neighbours,
                            knn=self.knn,
                            use_touching_neighbours_3d=self.use_touching_neighbours_3d,
                            touching_dilation_um=self.touching_dilation_um,
                        )

                    phenotype_column = None
                    if self.do_phenotype_calling:
                        phenotype_column = (
                            f"phenotype_{self.phenotype_1}_vs_{self.phenotype_2}"
                        )

                    if self.calculate_phenotype_similarity_knn:
                        if not self.use_knn_neighbours or self.knn is None:
                            print(
                                "Skipping KNN phenotype similarity score: KNN neighbours are disabled."
                            )
                        else:
                            print(
                                "Calculating phenotype similarity ratio based on KNN neighbours..."
                            )
                            add_phenotype_similarity(
                                i,
                                neighbors_column=f"neighbours_{self.knn}_KNN",
                                output_column=f"phenotype_similarity_ratio_{self.knn}_KNN",
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
                        split_phenotype_mask(i, self.phenotype_1, self.phenotype_2)
                    print(
                        f"Finished processing sample {i}\n{idx + 1}/{len(self.sample_path_list)}"
                    )
                except Exception as sample_error:
                    import traceback

                    traceback.print_exc()
                    failed_samples.append(f"{i}: {sample_error}")
                    print(f"Skipping sample due to error: {i}")
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
