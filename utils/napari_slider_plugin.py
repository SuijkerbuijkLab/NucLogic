"""
Warning: This plugin is vibecoded using claude
Interactive property-based filtering of segmentation labels in napari.

This adds a dock widget that lets you filter which cells (labels) are shown in a
napari ``Labels`` layer based on the statistics stored in the pipeline's
``{sample}_properties.tsv`` file. You pick one or more numeric properties, each
gets a histogram + range slider, and the labels layer updates in place so only
cells whose statistics fall inside every active range remain visible.

From the widget you can also:
  * split the current selection into a new labels layer,
  * save the current selection to a compressed TIFF (same metadata/compression
    as the rest of the pipeline),
  * write the inside/outside annotation back into the properties TSV as a new
    column.

Integration (see ``PySide6/app.py``)::

    from utils.napari_slider_plugin import attach_property_filter
    ...
    attach_property_filter(viewer)

For each segmentation ``Labels`` layer, set its properties file (and, ideally,
voxel size) on the layer so the widget can find them without relying on
``layer.source.path`` (segmentation layers are built from arrays and have no
source path)::

    viewer.add_labels(
        seg, name=...,
        metadata={"properties_path": tsv_path, "voxel_size": (z, y, x)},
    )

Reopening the widget after closing it: press ``Shift-F`` in the viewer, or
right-click the napari menu/dock area and toggle it from the context menu.
"""

from __future__ import annotations

import os

# Match the host app's Qt binding (PySide6) *before* importing qtpy. If qtpy were
# to resolve to a different binding (e.g. PyQt5) than napari/the app, loading two
# Qt bindings in one process crashes hard. See app.py for the same pin.
os.environ.setdefault("QT_API", "pyside6")

from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from superqt import QLabeledDoubleRangeSlider

# Columns that identify a cell / are not meaningful to filter on with a slider.
_NON_FILTERABLE = {"label", "bounding_box", "timepoint", "time"}


def _safe_div(a, b):
    """a / b with 0-division -> NaN instead of inf (so it's dropped by filters)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = a / b
    return np.where(np.isfinite(out), out, np.nan)


def _safe_log10(a):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log10(a)
    return np.where(np.isfinite(out), out, np.nan)


def _safe_log2(a):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log2(a)
    return np.where(np.isfinite(out), out, np.nan)


# Derived-column operations. ``binary`` marks whether a second column (B) is used.
# Each entry: label -> (needs_B, function(a, b)).
_OPERATIONS = {
    "A / B": (True, lambda a, b: _safe_div(a, b)),
    "log10(A / B)": (True, lambda a, b: _safe_log10(_safe_div(a, b))),
    "log2(A / B)": (True, lambda a, b: _safe_log2(_safe_div(a, b))),
    "A - B": (True, lambda a, b: a - b),
    "A + B": (True, lambda a, b: a + b),
    "A * B": (True, lambda a, b: a * b),
    "log10(A)": (False, lambda a, b: _safe_log10(a)),
    "log2(A)": (False, lambda a, b: _safe_log2(a)),
    "sqrt(A)": (False, lambda a, b: _safe_div(np.sqrt(np.abs(a)) * np.sign(a), 1)),
}

# Prefix used to auto-name a derived column, following the pipeline's convention
# (e.g. log_ratio_edu_..._dapi_...). Mirrors the keys of _OPERATIONS.
_NAME_PREFIX = {
    "A / B": "ratio",
    "log10(A / B)": "log_ratio",
    "log2(A / B)": "log2_ratio",
    "A - B": "diff",
    "A + B": "sum",
    "A * B": "product",
    "log10(A)": "log",
    "log2(A)": "log2",
    "sqrt(A)": "sqrt",
}
# Debounce (ms) between a slider moving and the labels being recomputed. Keeps
# dragging smooth on large volumes instead of recomputing on every pixel of drag.
_DEBOUNCE_MS = 150


def find_properties_tsv(layer) -> Path | None:
    """Locate the properties TSV for a labels layer.

    Prefers ``layer.metadata['properties_path']`` (set by the app when it adds
    the segmentation). Falls back to ``{image_stem}_properties.tsv`` next to the
    layer's source file if the layer was loaded from disk.
    """
    meta_path = layer.metadata.get("properties_path")
    if meta_path:
        p = Path(meta_path)
        if p.exists():
            return p

    src = getattr(layer.source, "path", None)
    if src is None:
        return None
    img_path = Path(src)
    # image.tif -> image_properties.tsv ; image_segmented.tif -> image_properties.tsv
    stem = img_path.stem
    for suffix in ("_segmented", "_cropped", ""):
        if suffix and stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    candidate = img_path.with_name(stem + "_properties.tsv")
    return candidate if candidate.exists() else None


def _numeric_filterable_columns(df: pd.DataFrame) -> list[str]:
    """Numeric columns worth exposing as filter sliders."""
    cols = []
    for col in df.columns:
        if col in _NON_FILTERABLE:
            continue
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].notna().any():
            cols.append(col)
    return cols


class _FilterRow(QFrame):
    """One property: wrapped name, histogram, double range slider, remove button."""

    def __init__(self, column: str, series: pd.Series, on_change, on_remove):
        super().__init__()
        self.column = column
        self.setFrameShape(QFrame.StyledPanel)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        values = series.to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        vmin = float(values.min()) if values.size else 0.0
        vmax = float(values.max()) if values.size else 1.0
        if vmin == vmax:  # avoid a zero-width slider
            vmax = vmin + 1.0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(2)

        # --- Header: wrapped name + remove button --------------------------
        header = QHBoxLayout()
        name_label = QLabel(column)
        name_label.setWordWrap(True)  # long names wrap instead of overflowing
        name_label.setStyleSheet("font-weight: bold;")
        name_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        header.addWidget(name_label, 1)
        remove_btn = QPushButton("✕")
        remove_btn.setFixedWidth(24)
        remove_btn.setToolTip("Remove this filter")
        remove_btn.clicked.connect(lambda: on_remove(self))
        header.addWidget(remove_btn, 0, Qt.AlignTop)
        layout.addLayout(header)

        # --- Histogram (optional; degrades gracefully) ---------------------
        self._span = None
        self._ax = None
        self._canvas = None
        self._add_histogram(layout, values, vmin, vmax)

        # --- Range slider ---------------------------------------------------
        self.slider = QLabeledDoubleRangeSlider(Qt.Horizontal)
        self.slider.setRange(vmin, vmax)
        self.slider.setValue((vmin, vmax))
        span = vmax - vmin
        self.slider.setSingleStep(span / 1000.0 if span else 1.0)
        self._on_change = on_change
        self.slider.valueChanged.connect(self._on_slider)
        layout.addWidget(self.slider)

    def _add_histogram(self, layout, values, vmin, vmax):
        if values.size == 0:
            return
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
        except Exception:
            return  # matplotlib unavailable -> just no histogram

        fig = Figure(figsize=(3, 0.8), tight_layout=True)
        canvas = FigureCanvasQTAgg(fig)
        canvas.setFixedHeight(80)
        canvas.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        ax = fig.add_subplot(111)
        ax.hist(values, bins=min(50, max(10, values.size // 20)), color="#5b9bd5")
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=6)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.margins(x=0)
        self._span = ax.axvspan(vmin, vmax, color="orange", alpha=0.25)
        self._ax = ax
        self._canvas = canvas
        layout.addWidget(canvas)

    def _update_histogram_span(self, lo, hi):
        if self._span is None or self._ax is None or self._canvas is None:
            return
        try:
            self._span.remove()
            self._span = self._ax.axvspan(lo, hi, color="orange", alpha=0.25)
            self._canvas.draw_idle()
        except Exception:
            pass

    def _on_slider(self, _=None):
        lo, hi = self.bounds()
        self._update_histogram_span(lo, hi)
        self._on_change()

    def bounds(self) -> tuple[float, float]:
        lo, hi = self.slider.value()
        return float(lo), float(hi)


class PropertyFilterWidget(QWidget):
    """Dock widget: filter a Labels layer's cells by their TSV statistics."""

    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self._rows: list[_FilterRow] = []
        self._active_layer_name = None

        # Debounce timer so dragging a slider doesn't recompute on every tick.
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(_DEBOUNCE_MS)
        self._debounce.timeout.connect(self._apply)

        # The last name we auto-generated, so we only overwrite an auto name and
        # never clobber one the user typed themselves.
        self._auto_name = ""

        root = QVBoxLayout(self)

        # --- Layer selection ------------------------------------------------
        root.addWidget(QLabel("Segmentation layer:"))
        self.layer_combo = QComboBox()
        self.layer_combo.currentIndexChanged.connect(self._on_layer_changed)
        root.addWidget(self.layer_combo)

        self.load_btn = QPushButton("Load properties")
        self.load_btn.clicked.connect(self._load_properties)
        root.addWidget(self.load_btn)

        # --- Section: compute a new column ---------------------------------
        calc_group = QGroupBox("Compute new column")
        calc_layout = QVBoxLayout(calc_group)

        calc_expr_row = QHBoxLayout()
        self.calc_a = QComboBox()
        self.calc_a.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.calc_a.currentTextChanged.connect(self._refresh_suggested_name)
        self.calc_op = QComboBox()
        self.calc_op.addItems(list(_OPERATIONS.keys()))
        self.calc_op.currentTextChanged.connect(self._on_calc_op_changed)
        self.calc_b = QComboBox()
        self.calc_b.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.calc_b.currentTextChanged.connect(self._refresh_suggested_name)
        calc_expr_row.addWidget(self.calc_a, 1)
        calc_expr_row.addWidget(self.calc_op, 0)
        calc_expr_row.addWidget(self.calc_b, 1)
        calc_layout.addLayout(calc_expr_row)

        calc_name_row = QHBoxLayout()
        calc_name_row.addWidget(QLabel("Name:"))
        self.calc_name = QLineEdit()
        self.calc_name.setPlaceholderText("auto-generated from A / op / B")
        calc_name_row.addWidget(self.calc_name, 1)
        calc_layout.addLayout(calc_name_row)

        calc_btn_row = QHBoxLayout()
        self.calc_save_chk = QCheckBox("Also save to TSV")
        calc_btn_row.addWidget(self.calc_save_chk)
        calc_btn_row.addStretch()
        self.calc_btn = QPushButton("Add column")
        self.calc_btn.clicked.connect(self._add_computed_column)
        calc_btn_row.addWidget(self.calc_btn)
        calc_layout.addLayout(calc_btn_row)
        root.addWidget(calc_group)

        # --- Section: filter cells -----------------------------------------
        filter_group = QGroupBox("Filter cells")
        filter_layout = QVBoxLayout(filter_group)

        add_row = QHBoxLayout()
        self.prop_combo = QComboBox()
        self.prop_combo.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        add_row.addWidget(self.prop_combo, 1)
        self.add_btn = QPushButton("Add filter")
        self.add_btn.clicked.connect(self._add_current_property)
        add_row.addWidget(self.add_btn)
        filter_layout.addLayout(add_row)

        self.filters_container = QWidget()
        self.filters_layout = QVBoxLayout(self.filters_container)
        self.filters_layout.setAlignment(Qt.AlignTop)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        # Force content to fit the width so long names wrap instead of scrolling off.
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(self.filters_container)
        filter_layout.addWidget(scroll, 1)

        self.reset_btn = QPushButton("Reset (show all)")
        self.reset_btn.clicked.connect(self._reset)
        filter_layout.addWidget(self.reset_btn)
        root.addWidget(filter_group, 1)

        # --- Section: export the selection ---------------------------------
        export_group = QGroupBox("Export selection")
        export_layout = QHBoxLayout(export_group)
        self.split_btn = QPushButton("Split → new layer")
        self.split_btn.setToolTip("Copy the current selection into a new labels layer")
        self.split_btn.clicked.connect(self._split_to_new_layer)
        export_layout.addWidget(self.split_btn)
        self.save_tiff_btn = QPushButton("Save selection to TIFF…")
        self.save_tiff_btn.clicked.connect(self._save_to_tiff)
        export_layout.addWidget(self.save_tiff_btn)
        root.addWidget(export_group)

        # --- Section: write the selection back into the TSV ----------------
        annotate_group = QGroupBox("Save selection to TSV column")
        annotate_layout = QVBoxLayout(annotate_group)
        ann_row1 = QHBoxLayout()
        ann_row1.addWidget(QLabel("Column:"))
        self.ann_col = QLineEdit()
        self.ann_col.setPlaceholderText("e.g. selected")
        ann_row1.addWidget(self.ann_col, 1)
        annotate_layout.addLayout(ann_row1)
        ann_row2 = QHBoxLayout()
        ann_row2.addWidget(QLabel("Inside:"))
        self.ann_inside = QLineEdit("True")
        ann_row2.addWidget(self.ann_inside)
        ann_row2.addWidget(QLabel("Outside:"))
        self.ann_outside = QLineEdit("False")
        ann_row2.addWidget(self.ann_outside)
        annotate_layout.addLayout(ann_row2)
        self.annotate_btn = QPushButton("Write column to TSV")
        self.annotate_btn.clicked.connect(self._annotate_tsv)
        annotate_layout.addWidget(self.annotate_btn)
        root.addWidget(annotate_group)

        # --- Status ---------------------------------------------------------
        self.status = QLabel("")
        self.status.setWordWrap(True)
        root.addWidget(self.status)

        self._set_controls_enabled(False)

        # Keep the layer list in sync with the viewer.
        self.viewer.layers.events.inserted.connect(self._refresh_layers)
        self.viewer.layers.events.removed.connect(self._refresh_layers)
        self._refresh_layers()

    @staticmethod
    def _hline():
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        return line

    # ------------------------------------------------------------------ layers
    def _labels_layers(self):
        from napari.layers import Labels

        return [ly for ly in self.viewer.layers if isinstance(ly, Labels)]

    def _refresh_layers(self, event=None):
        try:
            self._refresh_layers_impl()
        except RuntimeError:
            # This widget's Qt objects were destroyed (dock closed & recreated);
            # detach so we stop reacting to viewer events.
            for ev in (
                self.viewer.layers.events.inserted,
                self.viewer.layers.events.removed,
            ):
                try:
                    ev.disconnect(self._refresh_layers)
                except Exception:
                    pass

    def _refresh_layers_impl(self):
        current = self.layer_combo.currentText()
        self.layer_combo.blockSignals(True)
        self.layer_combo.clear()
        for ly in self._labels_layers():
            self.layer_combo.addItem(ly.name)
        # Try to keep the previous selection.
        idx = self.layer_combo.findText(current)
        if idx >= 0:
            self.layer_combo.setCurrentIndex(idx)
        self.layer_combo.blockSignals(False)
        self._on_layer_changed()

    def _current_layer(self):
        name = self.layer_combo.currentText()
        if not name or name not in self.viewer.layers:
            return None
        return self.viewer.layers[name]

    def _on_layer_changed(self, *_):
        layer = self._current_layer()
        name = layer.name if layer is not None else None
        # Only reset the active filters when the selected layer actually changes,
        # not on every layer-list refresh (e.g. when we add a split-off layer).
        if name != self._active_layer_name:
            self._clear_filters()
            self._active_layer_name = name
        has_props = layer is not None and "_properties_df" in layer.metadata
        self._set_controls_enabled(has_props)
        if has_props:
            self._populate_property_combo(layer.metadata["_properties_df"])
            self.status.setText("Properties already loaded for this layer.")
        else:
            self.prop_combo.clear()
            tsv = find_properties_tsv(layer) if layer is not None else None
            if tsv is None:
                self.status.setText(
                    "No properties loaded. No *_properties.tsv found for this layer."
                )
            else:
                self.status.setText(f"Found {tsv.name}. Click 'Load properties'.")

    # -------------------------------------------------------------- properties
    def _load_properties(self):
        layer = self._current_layer()
        if layer is None:
            return
        tsv = find_properties_tsv(layer)
        if tsv is None:
            self.status.setText("No matching *_properties.tsv found next to the image.")
            return
        df = pd.read_csv(tsv, sep="\t")
        if "label" not in df.columns:
            self.status.setText(f"{tsv.name} has no 'label' column; cannot filter.")
            return

        layer.metadata["_properties_df"] = df
        layer.metadata["_properties_path"] = str(tsv)
        # Cache a pristine copy of the labels so filtering is non-destructive.
        if "_original_data" not in layer.metadata:
            layer.metadata["_original_data"] = np.asarray(layer.data).copy()
        try:
            layer.features = df
        except Exception:
            pass  # features are convenience only; filtering uses the cached df

        self._populate_property_combo(df)
        self._set_controls_enabled(True)
        self.status.setText(
            f"Loaded {tsv.name}: {len(df)} rows, "
            f"{len(_numeric_filterable_columns(df))} filterable properties."
        )

    def _populate_property_combo(self, df: pd.DataFrame):
        cols = _numeric_filterable_columns(df)
        for combo in (self.prop_combo, self.calc_a, self.calc_b):
            prev = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(cols)
            idx = combo.findText(prev)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            combo.blockSignals(False)
        self._on_calc_op_changed(self.calc_op.currentText())

    def _on_calc_op_changed(self, op_label: str):
        needs_b = _OPERATIONS.get(op_label, (True, None))[0]
        self.calc_b.setEnabled(needs_b)
        self._refresh_suggested_name()

    def _suggested_name(self) -> str:
        """Auto-name a derived column following the pipeline naming scheme."""
        op_label = self.calc_op.currentText()
        needs_b, _ = _OPERATIONS.get(op_label, (True, None))
        prefix = _NAME_PREFIX.get(op_label, "derived")
        a = self.calc_a.currentText()
        if not a:
            return ""
        if needs_b:
            b = self.calc_b.currentText()
            if not b:
                return ""
            return f"{prefix}_{a}_{b}"
        return f"{prefix}_{a}"

    def _refresh_suggested_name(self, *_):
        """Fill the name box with the suggestion unless the user typed their own."""
        suggestion = self._suggested_name()
        current = self.calc_name.text()
        if current == "" or current == self._auto_name:
            self.calc_name.setText(suggestion)
        self._auto_name = suggestion

    # ------------------------------------------------------------------ filters
    def _add_current_property(self):
        layer = self._current_layer()
        if layer is None or "_properties_df" not in layer.metadata:
            return
        column = self.prop_combo.currentText()
        if not column:
            return
        if any(r.column == column for r in self._rows):
            self.status.setText(f"'{column}' is already being filtered.")
            return
        df = layer.metadata["_properties_df"]
        row = _FilterRow(
            column, df[column], on_change=self._schedule_apply, on_remove=self._remove_filter
        )
        self._rows.append(row)
        self.filters_layout.addWidget(row)
        self._apply()

    def _remove_filter(self, row: _FilterRow):
        if row in self._rows:
            self._rows.remove(row)
        row.setParent(None)
        row.deleteLater()
        self._apply()

    def _clear_filters(self):
        for row in self._rows:
            row.setParent(None)
            row.deleteLater()
        self._rows = []

    # ------------------------------------------------------------------- apply
    def _schedule_apply(self):
        self._debounce.start()

    def _compute_keep_mask(self, df: pd.DataFrame) -> np.ndarray:
        """Boolean mask over df rows: True where every active filter passes."""
        keep_mask = np.ones(len(df), dtype=bool)
        for row in self._rows:
            lo, hi = row.bounds()
            col = df[row.column].to_numpy(dtype=float)
            keep_mask &= (col >= lo) & (col <= hi)
        return keep_mask

    def _build_filtered(self, df, original, keep_mask):
        """Return a copy of ``original`` with only kept labels retained."""
        kept = df.loc[keep_mask]
        # Per-timepoint filtering: labels reset each timepoint, so a label id is
        # only unique within its own frame.
        if "timepoint" in df.columns and original.ndim == 4:
            filtered = np.zeros_like(original)
            for t in range(original.shape[0]):
                labels_t = kept.loc[kept["timepoint"] == t, "label"].to_numpy()
                filtered[t] = self._mask_slice(original[t], labels_t)
        else:
            filtered = self._mask_slice(original, kept["label"].to_numpy())
        return filtered

    def _apply(self):
        self._debounce.stop()
        layer = self._current_layer()
        if layer is None or "_properties_df" not in layer.metadata:
            return
        df = layer.metadata["_properties_df"]
        original = layer.metadata["_original_data"]
        keep_mask = self._compute_keep_mask(df)
        layer.data = self._build_filtered(df, original, keep_mask)
        n = int(keep_mask.sum())
        self.status.setText(
            f"Showing {n} / {len(df)} cells "
            f"({len(self._rows)} active filter{'s' if len(self._rows) != 1 else ''})."
        )

    @staticmethod
    def _mask_slice(arr: np.ndarray, keep_labels: np.ndarray) -> np.ndarray:
        """Zero out every label in ``arr`` that is not in ``keep_labels``.

        Uses a boolean lookup table indexed by label value, which is much faster
        than ``np.isin`` for the dense integer label ids the pipeline produces.
        """
        if keep_labels.size == 0:
            return np.zeros_like(arr)
        max_label = int(arr.max())
        lut = np.zeros(max_label + 1, dtype=bool)
        keep_labels = keep_labels[keep_labels <= max_label].astype(np.intp)
        lut[keep_labels] = True
        return np.where(lut[arr], arr, 0)

    # ------------------------------------------------------------------- reset
    def _reset(self):
        self._clear_filters()
        layer = self._current_layer()
        if layer is not None and "_original_data" in layer.metadata:
            layer.data = layer.metadata["_original_data"].copy()
            df = layer.metadata.get("_properties_df")
            n = len(df) if df is not None else "?"
            self.status.setText(f"Reset: showing all {n} cells.")

    # ------------------------------------------------------------- export/save
    def _current_selection_array(self, layer):
        """The currently filtered label array (a fresh copy) and its cell count."""
        df = layer.metadata["_properties_df"]
        original = layer.metadata["_original_data"]
        keep_mask = self._compute_keep_mask(df)
        return self._build_filtered(df, original, keep_mask), int(keep_mask.sum())

    def _split_to_new_layer(self):
        layer = self._current_layer()
        if layer is None or "_properties_df" not in layer.metadata:
            self.status.setText("Load properties first.")
            return
        data, n = self._current_selection_array(layer)
        new = self.viewer.add_labels(
            data,
            name=f"{layer.name} (filtered)",
            scale=layer.scale,
            blending=layer.blending,
            rendering=getattr(layer, "rendering", "iso_categorical"),
            metadata={
                k: layer.metadata.get(k)
                for k in ("sample", "voxel_size", "properties_path")
                if k in layer.metadata
            },
        )
        new.metadata["_original_data"] = data.copy()
        self.status.setText(f"Created '{new.name}' with {n} cells.")

    def _voxel_size(self, layer):
        vs = layer.metadata.get("voxel_size")
        if vs is not None and len(vs) >= 3:
            return tuple(float(v) for v in vs[-3:])
        scale = np.asarray(layer.scale, dtype=float)
        if scale.size >= 3:
            return tuple(scale[-3:])
        return (1.0, 1.0, 1.0)

    def _time_interval(self, df):
        if {"time", "timepoint"}.issubset(df.columns):
            nonzero = df.loc[df["timepoint"] > 0]
            if len(nonzero):
                ratios = nonzero["time"] / nonzero["timepoint"]
                val = float(ratios.median())
                if np.isfinite(val):
                    return val
        return 1.0

    def _save_to_tiff(self):
        layer = self._current_layer()
        if layer is None or "_properties_df" not in layer.metadata:
            self.status.setText("Load properties first.")
            return
        sample = layer.metadata.get("sample", "selection")
        default_name = f"{sample}_segmented_filtered.tif"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save selection to TIFF", default_name, "TIFF (*.tif *.tiff)"
        )
        if not path:
            return

        data, n = self._current_selection_array(layer)
        vz, vy, vx = self._voxel_size(layer)
        axes = "TZYX" if data.ndim == 4 else "ZYX"
        df = layer.metadata["_properties_df"]
        try:
            tifffile.imwrite(
                path,
                data,
                bigtiff=True,
                imagej=True,
                resolution=(1 / vx, 1 / vy),
                metadata={
                    "unit": "um",
                    "axes": axes,
                    "spacing": vz,
                    "finterval": self._time_interval(df),
                    "tunit": "h",
                },
                compression="zlib",
                compressionargs={"level": 8},
            )
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))
            self.status.setText(f"Save failed: {e}")
            return
        self.status.setText(f"Saved {n} cells to {Path(path).name}")

    # --------------------------------------------------------- derived columns
    def _add_computed_column(self):
        layer = self._current_layer()
        if layer is None or "_properties_df" not in layer.metadata:
            self.status.setText("Load properties first.")
            return
        name = self.calc_name.text().strip()
        if not name:
            self.status.setText("Enter a name for the new column.")
            return
        df = layer.metadata["_properties_df"]
        if name in df.columns:
            reply = QMessageBox.question(
                self,
                "Overwrite column?",
                f"Column '{name}' already exists. Overwrite it?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        op_label = self.calc_op.currentText()
        needs_b, func = _OPERATIONS[op_label]
        col_a = self.calc_a.currentText()
        if not col_a:
            self.status.setText("Select column A.")
            return
        a = df[col_a].to_numpy(dtype=float)
        b = None
        if needs_b:
            col_b = self.calc_b.currentText()
            if not col_b:
                self.status.setText("Select column B for this operation.")
                return
            b = df[col_b].to_numpy(dtype=float)

        result = np.asarray(func(a, b), dtype=float)
        df[name] = result

        try:
            layer.features = df
        except Exception:
            pass

        saved_note = ""
        if self.calc_save_chk.isChecked():
            tsv_path = layer.metadata.get("_properties_path") or layer.metadata.get(
                "properties_path"
            )
            if tsv_path and Path(tsv_path).exists():
                layer.metadata["_properties_df"] = self._write_column_to_disk(
                    tsv_path, df, name, df[name].to_numpy()
                )
                df = layer.metadata["_properties_df"]
                saved_note = " (saved to TSV)"
            else:
                saved_note = " (could not find TSV to save)"

        self._populate_property_combo(df)
        n_valid = int(np.isfinite(result).sum())
        self.status.setText(
            f"Added column '{name}' = {op_label}"
            f"{' of ' + col_a if not needs_b else f' ({col_a}, {self.calc_b.currentText()})'}; "
            f"{n_valid}/{len(result)} finite values{saved_note}."
        )

    def _write_column_to_disk(self, tsv_path, df_mem, colname, values):
        """Add ``colname`` to the on-disk TSV, aligning by row order or by keys.

        ``values`` must be aligned to ``df_mem`` rows. Reads the file fresh so
        columns added to disk since loading are preserved. Returns the new
        on-disk DataFrame.
        """
        df_disk = pd.read_csv(tsv_path, sep="\t")
        if len(df_disk) == len(df_mem):
            df_disk[colname] = values
        else:
            key_to_val = dict(zip(self._row_keys(df_mem), values))
            df_disk[colname] = [
                key_to_val.get(self._row_key(r), np.nan)
                for r in df_disk.itertuples(index=False)
            ]
        df_disk.to_csv(tsv_path, sep="\t", index=False)
        return df_disk

    # ------------------------------------------------------------- TSV annotate
    def _annotate_tsv(self):
        layer = self._current_layer()
        if layer is None or "_properties_df" not in layer.metadata:
            self.status.setText("Load properties first.")
            return
        col = self.ann_col.text().strip()
        if not col:
            self.status.setText("Enter a column name for the annotation.")
            return
        tsv_path = layer.metadata.get("_properties_path") or layer.metadata.get(
            "properties_path"
        )
        if not tsv_path or not Path(tsv_path).exists():
            self.status.setText("Could not locate the TSV file to write to.")
            return

        df_mem = layer.metadata["_properties_df"]
        if col in df_mem.columns:
            reply = QMessageBox.question(
                self,
                "Overwrite column?",
                f"Column '{col}' already exists in the TSV. Overwrite it?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        keep_mask = self._compute_keep_mask(df_mem)
        inside = self.ann_inside.text()
        outside = self.ann_outside.text()
        values = np.where(keep_mask, inside, outside)

        df_disk = self._write_column_to_disk(tsv_path, df_mem, col, values)
        layer.metadata["_properties_df"] = df_disk
        try:
            layer.features = df_disk
        except Exception:
            pass
        self.status.setText(
            f"Wrote column '{col}' to {Path(tsv_path).name} "
            f"({int(keep_mask.sum())} '{inside}', {int((~keep_mask).sum())} '{outside}')."
        )

    @staticmethod
    def _row_key(row):
        tp = getattr(row, "timepoint", 0)
        return (tp, getattr(row, "label"))

    @staticmethod
    def _row_keys(df):
        """Per-row (timepoint, label) keys aligned to df order."""
        tps = df["timepoint"] if "timepoint" in df.columns else np.zeros(len(df), int)
        return list(zip(tps, df["label"]))

    # ----------------------------------------------------------------- helpers
    def _set_controls_enabled(self, enabled: bool):
        for w in (
            self.prop_combo,
            self.add_btn,
            self.reset_btn,
            self.split_btn,
            self.save_tiff_btn,
            self.annotate_btn,
            self.calc_a,
            self.calc_op,
            self.calc_b,
            self.calc_btn,
            self.calc_save_chk,
        ):
            w.setEnabled(enabled)


def _find_menu(menubar, *keywords):
    """Return the top-level QMenu whose title matches one of ``keywords``."""
    for action in menubar.actions():
        menu = action.menu()
        if menu is None:
            continue
        title = action.text().replace("&", "").strip().lower()
        if any(k in title for k in keywords):
            return menu
    return None


def _add_reopen_menu_action(viewer, label, callback):
    """Add ``label`` under the napari Plugins menu (falls back to the menubar)."""
    try:
        main = getattr(viewer.window, "_qt_window", None)
        if main is None:
            return False
        menubar = main.menuBar()
        menu = _find_menu(menubar, "plugin")
        if menu is None:
            # Fall back: make our own top-level menu so it's still reachable.
            menu = menubar.addMenu("Property filter")
        else:
            menu.addSeparator()
        action = menu.addAction(label)
        action.triggered.connect(lambda *_: callback())
        return True
    except Exception:
        return False


def _ancestor_dock(widget):
    """Walk up from ``widget`` to its containing QDockWidget, if any."""
    from qtpy.QtWidgets import QDockWidget

    parent = widget.parent()
    while parent is not None:
        if isinstance(parent, QDockWidget):
            return parent
        parent = parent.parent()
    return None


def attach_property_filter(viewer, name: str = "Filter by properties"):
    """Create the widget, dock it, and make it reopenable.

    Registers an entry in the napari **Plugins** menu (and a ``Shift-F``
    shortcut) that reopens the widget after it has been closed. If a widget is
    still present it is raised (or shown, if hidden); otherwise a fresh one is
    created. Returns the widget.
    """

    def _existing_widget():
        main = getattr(viewer.window, "_qt_window", None)
        if main is None:
            return None
        for w in main.findChildren(PropertyFilterWidget):
            try:
                w.objectName()  # touch the C++ object; skip if it's been deleted
            except RuntimeError:
                continue
            return w
        return None

    def _open(*_):
        existing = _existing_widget()
        if existing is not None:
            dock = _ancestor_dock(existing)
            if dock is not None:
                dock.show()
                dock.raise_()
            else:
                existing.show()
            return existing
        widget = PropertyFilterWidget(viewer)
        viewer.window.add_dock_widget(widget, name=name, area="right")
        return widget

    widget = _open()

    try:
        viewer.bind_key("Shift-F", overwrite=True)(_open)
    except Exception:
        pass
    _add_reopen_menu_action(viewer, name, _open)

    return widget
