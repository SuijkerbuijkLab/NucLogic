import tkinter as tk
from tkinter import filedialog
import os


def launch_folder_picker():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    root.attributes("-topmost", True)  # Bring the dialog to the front

    selected_dirs = filedialog.askdirectory(
        title="Select Organism Directory", mustexist=True
    )

    if selected_dirs:
        with open("selected_path.txt", "w") as f:
            f.write(selected_dirs)
        print(f"Saved path: {selected_dirs}")
    else:
        print("No directory selected.")


if __name__ == "__main__":
    launch_folder_picker()
