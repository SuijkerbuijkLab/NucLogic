from server import server
from ui import app_ui
from pathlib import Path

from shiny import App

www_dir = Path(__file__).parent / "www"

app = App(app_ui, server, static_assets=www_dir)
