import tkfilebrowser

import tkinter as tk
from tkinter import ttk
import os


class dialog:

    # This is the window that will open when the program is run
    def __init__(self):
        # Creating the window
        self.root = tk.Tk()
        self.root.title("Organoid Segmenter")
        self.root.iconbitmap(r"./miscellaneous/organoid_segmenter.ico")
        self.dirs = []

        # Creating the intro text
        self.label = tk.Label(
            self.root,
            text="Select the directories containing your IMS files, each directory should have the IMS file of 1 organoid",
        ).pack(padx=10, pady=10)

        # Creating the button to select directories
        self.select_button = tk.Button(
            self.root, text="Select Directory", command=self.select_directory
        ).pack(padx=10, pady=10)

        # Showing the list of selected organoids
        self.list_organoids = tk.LabelFrame(
            self.root, text="You have selected the following organoids:"
        )
        self.list_organoids.pack(padx=10, pady=10, fill="both", expand=True)

        # Label to show if no organoids are selected
        self.no_organoids_label = tk.Label(
            self.list_organoids, text="No organoids selected yet."
        ).pack()

        # Frame for channel settings
        self.channel_frame = tk.LabelFrame(
            self.root,
            text="Channel Settings (Max 5), please name your nuclei channel 'Nuclei'",
        )
        self.channel_frame.pack(padx=10, pady=10, fill="both", expand=True)

        # Containers for entries and dropdowns
        self.channel_name_entries = []
        self.color_preview_labels = []

        # Create up to 5 channel input rows
        for i in range(5):
            row_frame = tk.Frame(self.channel_frame)
            row_frame.pack(padx=5, pady=5)

            tk.Label(row_frame, text=f"Channel {i+1} Name:").pack(
                side="left", padx=(0, 5)
            )
            name_entry = tk.Entry(row_frame, width=15)
            name_entry.pack(side="left")
            self.channel_name_entries.append(name_entry)

        # --- Advanced Settings Section ---
        self.advanced_frame = tk.LabelFrame(self.root, text="Advanced Settings")
        self.advanced_frame.pack(padx=10, pady=10, fill="both", expand=True)

        # Horizontal container for group boxes
        group_container = tk.Frame(self.advanced_frame)
        group_container.pack(padx=10, pady=10)

        # 🟦 Group 1: Existing Files
        self.cropped_exists_var = tk.BooleanVar(value=False)

        group1 = tk.LabelFrame(group_container, text="Existing Files")
        group1.pack(side="left", padx=10, fill="both", expand=True)

        tk.Checkbutton(
            group1,
            text="Folders already contain cropped tiff files",
            variable=self.cropped_exists_var,
        ).pack(anchor="w", pady=2)

        # 🟥 Group 2: Clean-up Options
        self.delete_cropped_var = tk.BooleanVar(value=True)
        self.delete_max_proj_var = tk.BooleanVar(value=True)
        self.delete_max_proj_tracked_var = tk.BooleanVar(value=True)

        group2 = tk.LabelFrame(group_container, text="Clean-up Options")
        group2.pack(side="left", padx=10, fill="both", expand=True)

        tk.Checkbutton(
            group2, text="Delete split cropped files", variable=self.delete_cropped_var
        ).pack(anchor="w", pady=2)
        tk.Checkbutton(
            group2, text="Delete max projection", variable=self.delete_max_proj_var
        ).pack(anchor="w", pady=2)
        tk.Checkbutton(
            group2,
            text="Delete tracked max projection",
            variable=self.delete_max_proj_tracked_var,
        ).pack(anchor="w", pady=2)

        # Creating the run button
        self.run_button = tk.Button(self.root, text="Run", command=self.run).pack(
            padx=10, pady=10
        )

    # This function handles the selection of directories
    def select_directory(self):
        # Asking the user to select directories
        selected = tkfilebrowser.askopendirnames(
            title="Select Directory",
            initialdir=".",
        )

        if selected:
            # Add selected dirs to the object
            self.dirs = list(selected)

            # Remove previously selected organoids from the list of selected organoids
            for label in self.list_organoids.winfo_children():
                label.destroy()

            # List the newly selected organoids
            for dir in self.dirs:
                organoid_label = tk.Label(
                    self.list_organoids, text=os.path.basename(dir)
                ).pack()

    # This function handles the run button click
    def run(self):
        # Check if any directories were selected
        if not self.dirs:
            tk.messagebox.showwarning(
                "No Directories Selected", "Please select at least one directory."
            )
            return
        # If directories were selected, print them and close the window
        else:
            self.channel_info = []
            for name_entry in self.channel_name_entries:
                name = name_entry.get().strip()
                self.channel_info.append(name)
            self.advanced_settings = {
                "cropped_exists": self.cropped_exists_var.get(),
                "delete_cropped": self.delete_cropped_var.get(),
                "delete_max_proj": self.delete_max_proj_var.get(),
                "delete_max_proj_tracked": self.delete_max_proj_tracked_var.get(),
            }
            self.root.destroy()


    def show(self):
        self.root.mainloop()
