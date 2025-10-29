import customtkinter as ctk
from gui import main_menu

ctk.set_appearance_mode("dark")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("green")  # Themes: "blue" (standard), "green", "dark-blue"

root = ctk.CTk()
root.title("Cassava Plant Disease Detector")
root.geometry("320x480")
root.grid_columnconfigure((0, 1, 2), weight=1)
root.grid_rowconfigure((0, 1, 2), weight=1)

main_menu(root)
root.mainloop()
