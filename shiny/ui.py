from shiny import ui


app_ui = ui.tags.html(
    ui.tags.head(
        ui.tags.link(rel="icon", href="organoid_segmenter.ico", type="image/x-icon")
    ),
    ui.page_navbar(
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
                    ui.layout_columns(
                        ui.div(
                            ui.h4("Run single cell segmentation:"),
                            ui.h5("Channel names:"),
                            ui.h6(
                                "Please select what each channel is and fill in a name for 'different' channels"
                            ),
                            *[
                                ui.div(
                                    ui.tags.span(
                                        f"{i+1}:",
                                        style="width: 30px; display: inline-block; margin-top: 5.5px;",
                                    ),
                                    ui.input_select(
                                        f"channel_name_{i}",
                                        "",
                                        choices=[
                                            "Nuclei",
                                            "WT",
                                            "CRC",
                                            "Different",
                                            "Empty",
                                        ],
                                        selected="Empty",
                                        width="150px",
                                    ),
                                    ui.input_text(
                                        f"channel_name_alternative_{i}",
                                        "",
                                        value="Alternative name",
                                        width="200px",
                                    ),
                                    style="margin-bottom: 0px; display: flex; align-items: right;",
                                )
                                for i in range(5)
                            ],
                            ui.h5("Advanced Settings"),
                            ui.tags.div(
                                ui.input_checkbox(
                                    "cropped_exists",
                                    "Folders already contain cropped tiff files",
                                    False,
                                ),
                                ui.input_checkbox(
                                    "delete_cropped", "Delete split cropped files", True
                                ),
                                ui.input_checkbox(
                                    "delete_max_proj", "Delete max projection", True
                                ),
                                ui.input_checkbox(
                                    "delete_max_proj_tracked",
                                    "Delete tracked max projection",
                                    True,
                                ),
                                ui.input_checkbox(
                                    "is_fixed", "Data is from fixed organoids", False
                                ),
                                style="font-size: 14px;",
                            ),
                            ui.input_action_button(
                                "run_segmenter", "Run Segmenter", width="200px"
                            ),
                            ui.output_ui("segmentation_progress"),
                        ),
                        ui.div(
                            ui.h4("Recalculate statistics experiment wide:"),
                            ui.h6(
                                "Select all organoids within this experiment that you want to use to recalculate phenotypes and knn_features",
                                style="margin-top: 5px;",
                            ),
                            ui.layout_columns(
                                ui.input_action_button(
                                    "recalculate_statistics",
                                    "Recalculate Statistics",
                                    width="250px",
                                ),
                                ui.output_ui("recalculate_stats_progress"),
                                style="margin-bottom: 10px; margin-top: 10px;",
                            ),
                            ui.tags.small(
                                "New statistics can be found in the properties_recalculated folder of every organoid"
                            ),
                        ),
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
                        ui.h4("Crop organoids into new IMS files:"),
                        ui.h6(
                            "Uses your exisiting IMS files to crop out the most center organoid and generate a new IMS file with similair metadata",
                            style="margin-top: 5px; margin-bottom: 15px;",
                        ),
                        ui.h5("Channel names:"),
                        ui.h6("Please name the channel containing nuclei 'Nuclei'"),
                        *[
                            ui.div(
                                ui.tags.span(
                                    f"{i+1}:",
                                    style="width: 30px; display: inline-block; margin-top: 5.5px;",
                                ),
                                ui.input_text(
                                    f"channel_cropper_{i}", "", width="200px"
                                ),
                                ui.tags.span(
                                    "Color:",
                                    style="margin-left: 10px;margin-right: 5px; display: inline-block; margin-top: 5.5px;",
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
                                    width="150px",
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
                            ui.input_action_button(
                                "run_cropper", "Run Cropper", width="200px"
                            ),
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
                ui.card_header("Plot Data"),
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.input_action_button(
                            "browse_dirs_plotting", "Select Data Directory"
                        ),
                        ui.input_action_button(
                            "toggle_select_plotting", "Select/Deselect All"
                        ),
                        ui.output_ui("organoid_list_plotting"),
                        ui.download_button(
                            "download_data", "Download data of selected organoids"
                        ),
                        bg="#f8f8f8",
                    ),
                    ui.div(
                        ui.input_switch(
                            "use_recalculated_statistics",
                            "Use recalculated statistics",
                            value=False,
                        ),
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
                        ui.h4("Visualize movies using Napari"),
                        ui.h6("Select what type of movies you want to visualize:"),
                        ui.tags.div(
                            ui.input_checkbox(
                                "segmentation_result",
                                "Segmentation result",
                                False,
                            ),
                            ui.input_checkbox("projXY", "Max Z projection", False),
                            ui.input_checkbox(
                                "projXY_tracked", "Tracked max Z projection", False
                            ),
                            ui.input_checkbox(
                                "full_movie",
                                "Full movie (takes longer to load)",
                                False,
                            ),
                            style="font-size: 14px;",
                        ),
                        ui.input_action_button("launch_napari", "Launch Napari"),
                    ),
                ),
            ),
        ),
        title=ui.tags.div(
            ui.tags.span(
                "Organoid Analyzer", style="font-weight: bold; font-size: 20px;"
            ),
            ui.tags.img(
                src="organoid_segmenter.ico",
                style=(
                    "position: absolute; top: -4px; left: 1212px; height: 30px;"
                    "z-index: 1000;"
                ),
                title="Organoid Segmenter",
            ),
            style="position: relative;",
        ),
    ),
)
