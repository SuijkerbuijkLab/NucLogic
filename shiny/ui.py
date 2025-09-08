from shiny import ui

app_ui = ui.page_navbar(
    ui.nav_panel(
        "Single cell segmentation",
        ui.card(
            ui.card_header("Single cell segmentation"),
            ui.layout_sidebar(
                ui.sidebar(
                    ui.input_action_button(
                        "browse_dirs_segmentation", "Select Data Directory"
                    ),
                    ui.input_action_button(
                        "toggle_select_segmentation", "Select/Deselect All"
                    ),
                    ui.output_ui("organoid_list_segmentation"),
                    bg="#f8f8f8",
                ),
                ui.div(
                    ui.h4("Channel names:"),
                    ui.h6("Please name the channel containing nuclei 'Nuclei'"),
                    *[
                        ui.tags.div(
                            ui.tags.span(
                                f"{i+1}:", style="width: 30px; display: inline-block;"
                            ),
                            ui.tags.span(
                                ui.input_text(f"channel_{i}", "", width="200px"),
                                style="display: inline-block;",
                            ),
                            style="margin-bottom: 0px;",
                        )
                        for i in range(5)
                    ],
                    ui.h4("Advanced Settings"),
                    ui.input_checkbox(
                        "cropped_exists",
                        "Folders already contain cropped tiff files",
                        False,
                    ),
                    ui.input_checkbox(
                        "delete_cropped", "Delete split cropped files", True
                    ),
                    ui.input_checkbox("delete_max_proj", "Delete max projection", True),
                    ui.input_checkbox(
                        "delete_max_proj_tracked", "Delete tracked max projection", True
                    ),
                    ui.input_action_button("run_segmenter", "Run Segmenter"),
                    ui.output_ui("segmentation_progress"),
                ),
            ),
        ),
    ),
    ui.nav_panel(
        "Organoid Cropper",
        ui.card(
            ui.card_header("Organoid Cropper"),
            ui.layout_sidebar(
                ui.sidebar(
                    ui.input_action_button(
                        "browse_dirs_cropper", "Select Data Directory"
                    ),
                    ui.input_action_button(
                        "toggle_select_cropper", "Select/Deselect All"
                    ),
                    ui.output_ui("organoid_list_cropper"),
                    bg="#f8f8f8",
                ),
                ui.div(
                    ui.h4("Channel names:"),
                    ui.h6("Please name the channel containing nuclei 'Nuclei'"),
                    *[
                        ui.div(
                            ui.tags.span(
                                f"{i+1}:", style="width: 30px; display: inline-block;"
                            ),
                            ui.input_text(f"channel_cropper_{i}", "", width="150px"),
                            ui.tags.span(
                                "Color:", style="margin-left: 10px;margin-right: 5px;"
                            ),
                            ui.input_select(
                                f"channel_color_cropper_{i}",
                                "",
                                choices=[
                                    "white",
                                    "red",
                                    "green",
                                    "blue",
                                    "cyan",
                                    "magenta",
                                    "yellow",
                                    "gray",
                                    "lime",
                                ],
                            ),
                            style="margin-bottom: 0px; display: flex; align-items: right;",
                        )
                        for i in range(5)
                    ],
                    ui.h4("Advanced Settings"),
                    ui.input_checkbox(
                        "cropped_exists_cropper",
                        "Folders already contain cropped tiff files",
                        False,
                    ),
                    ui.input_checkbox(
                        "delete_cropped_cropper", "Delete split cropped files", True
                    ),
                    ui.input_checkbox(
                        "delete_max_proj_cropper", "Delete max projection", True
                    ),
                    ui.input_checkbox(
                        "delete_max_proj_tracked_cropper",
                        "Delete tracked max projection",
                        True,
                    ),
                    ui.layout_columns(
                        ui.input_action_button("run_cropper", "Run Cropper"),
                        ui.output_ui("cropper_spinner"),
                    ),
                    ui.output_ui("cropper_progress"),
                ),
            ),
        ),
    ),
    ui.nav_panel(
        "Plot Data",
        ui.card(
            ui.card_header("Growth Plots"),
            ui.layout_sidebar(
                ui.sidebar(
                    ui.input_action_button(
                        "browse_dirs_plotting", "Select Data Directory"
                    ),
                    ui.input_action_button(
                        "toggle_select_plotting", "Select/Deselect All"
                    ),
                    ui.output_ui("organoid_list_plotting"),
                    bg="#f8f8f8",
                ),
                ui.div(
                    ui.input_numeric(
                        "ylim_max",
                        "Set Y-axis upper limit (optional)",
                        value=None,
                        min=0,
                        step=1,
                    ),
                    ui.output_plot("growth_by_sample_plot"),
                    ui.output_plot("growth_by_type_plot"),
                ),
            ),
        ),
    ),
    ui.nav_panel(
        "Visualize Napari",
        ui.card(
            ui.card_header("Visualize Napari"),
            ui.layout_sidebar(
                ui.sidebar(
                    ui.input_action_button(
                        "browse_dirs_napari", "Select Data Directory"
                    ),
                    ui.input_action_button(
                        "toggle_select_napari", "Select/Deselect All"
                    ),
                    ui.output_ui("organoid_list_napari"),
                    bg="#f8f8f8",
                ),
                ui.div(
                    ui.input_action_button("launch_napari", "Launch Napari"),
                ),
            ),
        ),
    ),
    title="Organoid Analyzer",
)
