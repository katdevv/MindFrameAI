import tkinter as tk
from tkinter import ttk, messagebox
import time
import psutil
import subprocess
import threading
from datetime import datetime, timedelta
from typing import List, Set
import os
import platform

APPLICATIONS = [
    "chrome.exe", "msedge.exe", "firefox.exe", "code.exe", "notepad.exe", "brave.exe", "opera.exe", "safari.exe",
    "youtube.exe", "spotify.exe", "discord.exe", "teams.exe", "zoom.exe", "word.exe", "excel.exe",
    "powerpoint.exe", "outlook.exe", "onenote.exe", "skype.exe", "telegram.exe", "whatsapp.exe", "twitch.exe",
    "twitter.exe", "facebook.exe", "instagram.exe",
    "linkedin.exe", "pinterest.exe", "tumblr.exe", "snapchat.exe", "tiktok.exe", "roblox.exe", "minecraft.exe",
    "fortnite.exe", "valorant.exe", "apexlegends.exe", "csgo.exe", "leagueoflegends.exe", "dota2.exe",
    "overwatch.exe", "pubg.exe", "warzone.exe"
]


class FocusMode:
    def __init__(self):
        self.blocked_apps: Set[str] = set()
        self.is_active: bool = False
        self.session_start_time: datetime = None
        self.session_duration: timedelta = None
        self.monitoring_thread: threading.Thread = None
        self.stop_monitoring: threading.Event = threading.Event()
        self.blocked_processes: List[int] = []

    def set_blocked_apps(self, apps: List[str]) -> None:
        self.blocked_apps = {app.lower() for app in apps}

    def start_focus_session(self, duration_minutes: int) -> bool:
        if self.is_active:
            return False

        self.is_active = True
        self.session_start_time = datetime.now()
        self.session_duration = timedelta(minutes=duration_minutes)
        self.stop_monitoring.clear()

        self.monitoring_thread = threading.Thread(target=self._monitor_applications)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        return True

    def stop_focus_session(self) -> None:
        if not self.is_active:
            return

        self.is_active = False
        self.stop_monitoring.set()
        self._restore_blocked_processes()

        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2)

    def get_session_info(self) -> dict:
        if not self.is_active:
            return {"status": "inactive"}

        elapsed = datetime.now() - self.session_start_time
        remaining = self.session_duration - elapsed

        return {
            "status": "active",
            "elapsed_minutes": int(elapsed.total_seconds() / 60),
            "remaining_minutes": max(0, int(remaining.total_seconds() / 60)),
            "blocked_apps": list(self.blocked_apps),
            "blocked_processes_count": len(self.blocked_processes)
        }

    def _monitor_applications(self) -> None:
        while not self.stop_monitoring.is_set():
            try:
                if datetime.now() >= self.session_start_time + self.session_duration:
                    self.stop_focus_session()
                    break

                for proc in psutil.process_iter(['pid', 'name']):
                    try:
                        process_name = proc.info['name'].lower()
                        if any(blocked_app.replace('.exe', '').lower() in process_name for blocked_app in
                               self.blocked_apps):
                            self._block_process(proc.info['pid'], process_name)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue

                time.sleep(2)

            except Exception as e:
                print(f"მონიტორინგის შეცდომა: {e}")
                time.sleep(5)

    def _block_process(self, pid: int, process_name: str) -> None:
        if pid in self.blocked_processes:
            return

        try:
            if platform.system() == "Windows":
                subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                               capture_output=True, check=False)
            else:
                os.kill(pid, 9)

            self.blocked_processes.append(pid)

        except Exception as e:
            print(f"პროცესის ბლოკირების შეცდომა {process_name}: {e}")

    def _restore_blocked_processes(self) -> None:
        self.blocked_processes.clear()


class FocusModeGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Focus Mode ")
        self.root.geometry("800x600")
        self.root.configure(bg='#f0f0f0')

        self.focus_mode = FocusMode()
        self.app_vars = {}
        self.chosen_applications = []

        self.setup_ui()
        self.update_timer()

    def setup_ui(self):
        # მთავარი ფრეიმი
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # სათაური
        title_label = tk.Label(main_frame, text="🎯 Focus Mode - ფოკუს მოუდი",
                               font=('Arial', 18, 'bold'), bg='#f0f0f0', fg='#2c3e50')
        title_label.pack(pady=(0, 20))

        # ძირითადი კონტენტი
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # მარცხენა ფანელი - აპლიკაციების არჩევა
        left_frame = ttk.LabelFrame(content_frame, text="აირჩიეთ ბლოკირებული აპლიკაციები", padding="10")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # სწრაფი არჩევის ღილაკები
        quick_select_frame = ttk.Frame(left_frame)
        quick_select_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(quick_select_frame, text="ყველას არჩევა",
                   command=self.select_all).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(quick_select_frame, text="ყველას გაუქმება",
                   command=self.deselect_all).pack(side=tk.LEFT, padx=(0, 5))

        # Scrollable frame ჩექბოქსებისთვის
        canvas = tk.Canvas(left_frame, height=300)
        scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # ჩექბოქსების შექმნა
        self.create_checkboxes(scrollable_frame)

        # მარჯვენა ფანელი - კონტროლები
        right_frame = ttk.LabelFrame(content_frame, text="სესიის კონტროლი", padding="10")
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))

        # დროის არჩევა
        time_frame = ttk.LabelFrame(right_frame, text="დროის ლიმიტი", padding="10")
        time_frame.pack(fill=tk.X, pady=(0, 10))

        # წინასწარ განსაზღვრული დროები
        self.time_var = tk.StringVar(value="30")
        time_options = [("15 წუთი", "15"), ("30 წუთი", "30"), ("45 წუთი", "45"),
                        ("1 საათი", "60"), ("2 საათი", "120")]

        for text, value in time_options:
            ttk.Radiobutton(time_frame, text=text, variable=self.time_var,
                            value=value).pack(anchor=tk.W)

        # კასტომ დრო
        custom_frame = ttk.Frame(time_frame)
        custom_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Radiobutton(custom_frame, text="სხვა:", variable=self.time_var,
                        value="custom").pack(side=tk.LEFT)
        self.custom_time_entry = ttk.Entry(custom_frame, width=5)
        self.custom_time_entry.pack(side=tk.LEFT, padx=(5, 0))
        ttk.Label(custom_frame, text="წუთი").pack(side=tk.LEFT, padx=(5, 0))

        # კონტროლის ღილაკები
        control_frame = ttk.LabelFrame(right_frame, text="კონტროლი", padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 10))

        self.start_button = tk.Button(control_frame, text="▶️ დაწყება",
                                      command=self.start_session, font=('Arial', 12, 'bold'),
                                      bg='#27ae60', fg='white', relief=tk.FLAT)
        self.start_button.pack(fill=tk.X, pady=(0, 5))

        self.stop_button = tk.Button(control_frame, text="⏹️ შეწყვეტა",
                                     command=self.stop_session, font=('Arial', 12, 'bold'),
                                     bg='#e74c3c', fg='white', relief=tk.FLAT, state=tk.DISABLED)
        self.stop_button.pack(fill=tk.X)

        # სტატუსის ფანელი
        status_frame = ttk.LabelFrame(right_frame, text="სტატუსი", padding="10")
        status_frame.pack(fill=tk.BOTH, expand=True)

        self.status_label = tk.Label(status_frame, text="სესია არ არის აქტიური",
                                     font=('Arial', 10), bg='#f0f0f0', fg='#7f8c8d')
        self.status_label.pack()

        self.timer_label = tk.Label(status_frame, text="",
                                    font=('Arial', 14, 'bold'), bg='#f0f0f0', fg='#2c3e50')
        self.timer_label.pack(pady=(10, 0))

        self.blocked_apps_label = tk.Label(status_frame, text="",
                                           font=('Arial', 9), bg='#f0f0f0', fg='#95a5a6',
                                           wraplength=200, justify=tk.LEFT)
        self.blocked_apps_label.pack(pady=(10, 0))

    def create_checkboxes(self, parent):
        # კატეგორიებად დაყოფა
        categories = {
            "🌐 ბრაუზერები": ["chrome.exe", "msedge.exe", "brave.exe"],
            "💻 სამუშაო აპლიკაციები": ["notepad.exe", "word.exe", "excel.exe", "powerpoint.exe", "outlook.exe"],
            "💬 კომუნიკაცია": ["discord.exe", "teams.exe", "zoom.exe", "skype.exe", "telegram.exe", "whatsapp.exe"],
            "🎵 მედია": ["youtube.exe", "spotify.exe"],
            "📱 სოციალური მედია": ["facebook.exe", "instagram.exe", "linkedin.exe", "pinterest.exe", "snapchat.exe",
                                  "tiktok.exe"],
            "🎮 გეიმები": ["roblox.exe", "minecraft.exe", "fortnite.exe", "valorant.exe", "csgo.exe",
                          "leagueoflegends.exe", "dota2.exe", "overwatch.exe", "callofduty.exe", "pubg.exe",
                          "warzone.exe", "fifa.exe"]
        }

        for category, apps in categories.items():

            cat_frame = ttk.LabelFrame(parent, text=category, padding="5")
            cat_frame.pack(fill=tk.X, pady=5)

            for app in apps:
                if app in APPLICATIONS:
                    var = tk.BooleanVar()
                    self.app_vars[app] = var

                    app_name = app.replace('.exe', '').title()
                    checkbox = ttk.Checkbutton(cat_frame, text=app_name, variable=var)
                    checkbox.pack(anchor=tk.W, padx=10)

    def select_all(self):
        for var in self.app_vars.values():
            var.set(True)

    def deselect_all(self):
        for var in self.app_vars.values():
            var.set(False)

    def select_social_media(self):
        social_apps = ["reddit.exe", "twitter.exe", "facebook.exe", "instagram.exe",
                       "linkedin.exe", "pinterest.exe", "tumblr.exe", "snapchat.exe", "tiktok.exe"]
        for app in social_apps:
            if app in self.app_vars:
                self.app_vars[app].set(True)

    def select_games(self):
        game_apps = ["roblox.exe", "minecraft.exe", "fortnite.exe", "valorant.exe",
                     "apexlegends.exe", "csgo.exe", "leagueoflegends.exe", "dota2.exe"]
        for app in game_apps:
            if app in self.app_vars:
                self.app_vars[app].set(True)

    def get_selected_apps(self):
        return [app for app, var in self.app_vars.items() if var.get()]

    def get_duration(self):
        if self.time_var.get() == "custom":
            try:
                return int(self.custom_time_entry.get())
            except ValueError:
                return 30
        return int(self.time_var.get())

    def start_session(self):
        selected_apps = self.get_selected_apps()

        if not selected_apps:
            messagebox.showwarning("გაფრთხილება", "მოგნიშნეთ მინიმუმ ერთი აპლიკაცია!")
            return

        duration = self.get_duration()

        if duration <= 0:
            messagebox.showwarning("გაფრთხილება", "მიუთითეთ სწორი დრო!")
            return

        self.focus_mode.set_blocked_apps(selected_apps)

        if self.focus_mode.start_focus_session(duration):
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.status_label.config(text="ფოკუს სესია აქტიურია 🔴", fg='#e74c3c')
            messagebox.showinfo("წარმატება", f"ფოკუს მოუდი დაიწყო {duration} წუთით!\n"
                                             f"ბლოკირებული აპლიკაციები: {len(selected_apps)}")

    def stop_session(self):
        self.focus_mode.stop_focus_session()
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="სესია შეწყდა", fg='#7f8c8d')
        self.timer_label.config(text="")
        self.blocked_apps_label.config(text="")
        messagebox.showinfo("შეტყობინება", "ფოკუს მოუდი შეწყდა!")

    def update_timer(self):
        if self.focus_mode.is_active:
            info = self.focus_mode.get_session_info()

            if info["status"] == "active":
                remaining = info["remaining_minutes"]
                self.timer_label.config(text=f"⏰ {remaining} წუთი")

                blocked_text = f"ბლოკირებული: {len(info['blocked_apps'])} აპლიკაცია"
                self.blocked_apps_label.config(text=blocked_text)

                if remaining <= 0:
                    self.stop_session()
            else:
                self.stop_session()

        self.root.after(1000, self.update_timer)  # 1 წამში ერთხელ განახლება

    def run(self):
        # Window-ის დახურვისას სესიის შეწყვეტა
        def on_closing():
            if self.focus_mode.is_active:
                result = messagebox.askyesno("დახურვა", "ფოკუს სესია აქტიურია. გნებავთ შეწყვეტა?")
                if result:
                    self.focus_mode.stop_focus_session()
                    self.root.destroy()
            else:
                self.root.destroy()

        self.root.protocol("WM_DELETE_WINDOW", on_closing)
        self.root.mainloop()


if __name__ == "__main__":
    app = FocusModeGUI()
    app.run()
