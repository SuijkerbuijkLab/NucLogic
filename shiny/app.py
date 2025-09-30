from server import server
from ui import app_ui

from shiny import App

app = App(app_ui, server)
