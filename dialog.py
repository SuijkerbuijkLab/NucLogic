import tkfilebrowser

import tkinter as tk
import os


class dialog:

    # This is the window that will open when the program is run
    def __init__(self):
        # Creating the window
        self.root = tk.Tk()
        self.root.title("Organoid Segmenter")
        self.root.iconbitmap(r"./organoid_segmenter.ico")
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

        # Creating the run button
        self.run_button = tk.Button(self.root, text="Run", command=self.run).pack(
            padx=10, pady=10
        )

        self.root.mainloop()

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
            print(f"Selected directories:{self.dirs}")
            self.root.destroy()
