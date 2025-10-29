import customtkinter as ctk
from PIL import Image, ImageTk
from camera import capture_frame, save_image
from preprocessing import preprocess_image
from model import predict, decode_predictions
from tkinter import END
import cv2

def main_menu(root):
    global main_frame
    main_frame = ctk.CTkFrame(root, width=310, height=470,corner_radius=10)
    main_frame.grid(row=0, column=0, columnspan=3, rowspan=3, padx=20, pady=20, sticky="nsew")
    main_frame.pack(pady=40)

    logo_label = ctk.CTkLabel(main_frame, text="Cassava Diseases Detection", font=("gill sans bold", 18, 'bold'))
    logo_label.place(relx=0.5, rely=0.5, anchor="center")
    logo_label.pack()
    
    start_capture_button = ctk.CTkButton(main_frame, text="Start Capture", width=470, height=100, font=("gill sans", 20, "bold"), command=lambda: show_live_feed(root))
    start_capture_button.place(relx=0.5, rely=0.25, anchor="center")
    start_capture_button.pack(padx=20,pady=30)

    exit_button = ctk.CTkButton(main_frame, text="Exit",width=470, height=100, font=("gill sans", 20, "bold"), command=root.quit)
    exit_button.place(relx=0.5, rely=0.40, anchor="center")
    exit_button.pack(padx=20,pady=30)

def show_live_feed(root):
    main_frame.pack_forget()
    global live_feed_frame
    live_feed_frame = ctk.CTkFrame(root, width=320, height=480)
    live_feed_frame.pack()

    global lmain
    lmain = ctk.CTkLabel(live_feed_frame,text="")
    lmain.pack(pady=10)

    capture_button = ctk.CTkButton(live_feed_frame, text="Capture",width=470, height=75, font=("gill sans", 10, "bold"), command=capture_image)
    capture_button.pack(pady=5)

    process_button = ctk.CTkButton(live_feed_frame, text="Process",width=470, height=75, font=("gill sans", 10, "bold"), command=lambda: process_image(root))
    process_button.pack(pady=5)

    return_button = ctk.CTkButton(live_feed_frame, text="Return",width=470, height=75, font=("gill sans", 10, "bold"), command=lambda: return_to_main(root))
    return_button.pack(pady=5)

    show_frame()

def show_frame():
    frame = capture_frame()

    # Draw a bounding box with size 180x180 at the center of the frame
    height, width, _ = frame.shape
    start_point = (width // 2 - 90, height // 2 - 90)  # Top-left corner of the bounding box
    end_point = (width // 2 + 90, height // 2 + 90)    # Bottom-right corner of the bounding box
    color = (0, 255, 0)  # Green color for the bounding box
    thickness = 2  # Thickness of the bounding box

    cv2.rectangle(frame, start_point, end_point, color, thickness)

    # Convert frame to display
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)


def capture_image():
    global captured_frame
    captured_frame = capture_frame()
    save_image(captured_frame, "/home/pi/project/env/save/captured_image.jpg")

def process_image(root):
    live_feed_frame.pack_forget()
    global process_frame
    process_frame = ctk.CTkFrame(root, width=320, height=480)
    process_frame.pack()

    img = Image.open("/home/pi/project/env/save/captured_image.jpg")
    img = img.resize((244, 244))
    img = ImageTk.PhotoImage(img)

    captured_image_label = ctk.CTkLabel(process_frame, image=img, text="")
    captured_image_label.image = img
    captured_image_label.pack(pady=10)

    findings_label = ctk.CTkLabel(process_frame, text="Findings")
    findings_label.pack(pady=10)

    # Preprocess the image and make predictions
    preprocessed_image = preprocess_image("/home/pi/project/env/save/captured_image.jpg")
    predictions = predict(preprocessed_image)

    # Decode predictions to human-readable format
    result = decode_predictions(predictions)
    findings = f"{result}\n"

    # Create and configure the textbox to display findings
    textbox = ctk.CTkTextbox(process_frame, width=320, height=100)
    textbox.configure(state="normal")
    textbox.delete("0.0", END)
    textbox.insert("0.0", "Findings and Recommendations\n")
    textbox.insert(END, findings)
    textbox.configure(state="disabled")
    textbox.pack(pady=5)

    return_button = ctk.CTkButton(process_frame, text="Return",width=470, height=50, font=("gill sans", 10, "bold"), command=lambda: return_to_main(root))
    return_button.pack(pady=2)

def return_to_main(root):
    if 'live_feed_frame' in globals():
        live_feed_frame.destroy()
    if 'process_frame' in globals():
        process_frame.destroy()
    main_menu(root)
