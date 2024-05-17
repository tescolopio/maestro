import tkinter as tk
from tkinter import ttk
import time
import json
from datetime import datetime, timedelta
from threading import Thread

class CompletionTracker(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Completion Token Tracker")
        self.geometry("800x600")
        self.resizable(True, True)

        self.models = {}
        self.load_models_from_file()
        self.create_widgets()
        self.update_timer = self.schedule_update(1)  # Update every second
        self.bind("<Configure>", self.on_resize)  # Update on window resize
        self.start_time = datetime.now()  # Set the initial start time

    def load_models_from_file(self):
        try:
            with open("models.json", "r") as file:
                models = json.load(file)
                for name, data in models.items():
                    self.add_model(name, data)
        except FileNotFoundError:
            pass

    def save_models_to_file(self):
        with open("models.json", "w") as file:
            json.dump(self.models, file)

    def add_model(self, name, data=None):
        if data is None:
            data = {
                "tokens": 0,
                "last_call": datetime.now(),
                "today_count": 0
            }
        self.models[name] = data
        self.update_model(name)

    def update_model(self, name, tokens=None, last_call=None, today_count=None):
        if tokens:
            self.models[name]["tokens"] = tokens
        if last_call:
            self.models[name]["last_call"] = last_call
        if today_count:
            self.models[name]["today_count"] = today_count
        self.update_table()

    def remove_model(self, name):
        del self.models[name]
        self.update_table()
        self.save_models_to_file()

    def on_resize(self, event):
        self.update_table()

    def create_widgets(self):
        # Create the table
        columns = ("name", "tokens", "last_call")
        self.table = tk.ttk.Treeview(self, columns=columns, show="headings")
        self.table.heading("name", text="Model Name")
        self.table.heading("tokens", text="Completion Tokens")
        self.table.heading("last_call", text="Last Call")
        self.table.pack(side="top", fill="both", expand=True)

        # Create the countdown timer
        self.timer_label = tk.Label(self, text="Time elapsed: 0s")
        self.timer_label.pack(side="top")

        # Create the progress bar
        self.progress_bar = tk.ttk.Progressbar(self, orient="horizontal", length=300, mode="determinate")
        self.progress_bar.pack(side="top", padx=10, pady=10)

        # Create the close and minimize buttons
        self.create_menu()
        self.update_table()
        self.update_timer_label()
        self.update_progress_bar()

    def update_table(self):
        self.table.delete(*self.table.get_children())
        for model in self.models.items():
            self.table.insert("", "end", text=model[0], values=(model[0], model[1]["tokens"], model[1]["last_call"]))

    def schedule_update(self, interval):
        def update():
            self.update_timer_label()
            self.update_progress_bar()
            self.after(int(interval * 1000), update)
        Thread(target=update).start()

    def update_timer_label(self):
        elapsed_time = datetime.now() - self.start_time
        self.timer_label.config(text=f"Time elapsed: {int(elapsed_time.total_seconds())}s")

    def update_progress_bar(self):
        total_calls = sum([model["today_count"] for model in self.models.values()])
        total_possible = sum([model["tokens"] for model in self.models.values()])
        self.progress_bar["value"] = int(total_calls / total_possible * 100)

    def create_menu(self):
        menu = tk.Menu(self)
        self.config(menu=menu)

        file_menu = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Save", command=self.save_models_to_file)
        file_menu.add_command(label="Exit", command=self.quit)

        help_menu = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.about)

    def about(self):
        # Implement the about functionality
        pass

if __name__ == "__main__":
    app = CompletionTracker()
    app.mainloop()
