from shiny import ui


app_ui = ui.page_fillable(
    ui.card(
        ui.card_header("Organoid Growth Dashboard"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_text("path", "Path where all data is stored"),
                ui.input_action_button("toggle_select", "Select/Deselect All"),
                ui.output_ui("organoid_list"),
                bg="#f8f8f8",
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
    )
)
