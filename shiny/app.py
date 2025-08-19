from shiny import App
from ui import app_ui
from server import server

app = App(app_ui, server)
