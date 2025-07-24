import tkinter as tk
from tkinter import ttk


class progressbar:
    def __init__(self, total_tasks):
        self.root = tk.Tk()
        self.root.title("Processing Organoids")
        self.root.geometry("400x100")
        self.root.resizable(False, False)

        self.label = tk.Label(self.root, text="Processing organoids...")
        self.label.pack(pady=10)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.root,
            variable=self.progress_var,
            maximum=total_tasks,
            length=300,
            mode="determinate",
        )
        self.progress_bar.pack(pady=5)

        self.root.update()  # Make sure the window appears immediately

    def update(self, current):
        percent = (current / self.progress_bar["maximum"]) * 100
        self.label.config(text=f"Progress: {int(percent)}%")
        self.progress_var.set(current)
        self.root.update_idletasks()

    def close(self):
        self.root.destroy()
