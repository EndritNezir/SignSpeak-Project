import tkinter as tk
from tkinter import Label, Frame
import cv2
from PIL import Image, ImageTk
import gesture_recognition
import speech

# Function to start the camera
def start_camera():
    global cap, video_label, description_label, back_button
    cap = cv2.VideoCapture(0)
    description_label.pack_forget()
    back_button.place(relx=0.85, rely=0.05)
    update_frame()

# Function to stop camera
def stop_camera():
    global cap, video_label, description_label, back_button
    cap.release()
    video_label.config(image="")  # Remove last frame
    description_label.pack(pady=20)
    back_button.place_forget()

# Function to update camera frames
def update_frame():
    global cap, video_label, gesture_label
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img = ImageTk.PhotoImage(img)
            video_label.config(image=img)
            video_label.image = img

            # Recognize gesture
            gesture = gesture_recognition.recognize_gesture(frame)
            if gesture:
                detected_text = gesture_recognition.gesture_dict.get(gesture, "Unknown Gesture")
                gesture_label.config(text=f"Detected: {detected_text}")
                speech.speak(detected_text)  # Convert to speech

        video_label.after(10, update_frame)

# Function to create a button card
def create_card(parent, text, command, description, emoji):
    card = Frame(parent, bg="#d9d9d9", padx=15, pady=10, bd=2, relief="ridge")
    card.pack(pady=10, fill="x", padx=10)

    btn = tk.Button(card, text=f"{emoji} {text}", command=command, font=("Arial", 12, "bold"),
                    fg="white", bg="#3b3e54", activebackground="#5a5f78", padx=15, pady=5, bd=0, relief="flat")
    btn.pack(pady=5)

    label = Label(card, text=description, font=("Arial", 10), bg="#d9d9d9", wraplength=180, justify="center")
    label.pack()

# Function to exit application
def exit_app():
    root.quit()

# Initialize main window
root = tk.Tk()
root.title("SignSpeak - Gesture to Speech")
root.geometry("800x500")
root.configure(bg="#f4f4f4")

# Sidebar on the Left
sidebar = Frame(root, bg="#3b3e54", width=200, height=500, relief="raised", bd=2)
sidebar.pack(side="left", fill="y")

# Title Label in Sidebar
title_label = Label(sidebar, text="SignSpeak", font=("Arial", 14, "bold"), fg="white", bg="#3b3e54")
title_label.pack(pady=40)

# Buttons in Sidebar
create_card(sidebar, "Start Camera", start_camera, "Capture real-time hand gestures with AI-powered recognition.", "📷")
create_card(sidebar, "Stop Camera", stop_camera, "Turn off the camera and return to description.", "⏹️")
create_card(sidebar, "Exit", exit_app, "Close the application safely.", "❌")

# Back Button (Right Side, Initially Hidden)
back_button = tk.Button(root, text="⬅️ Back", command=stop_camera, font=("Arial", 12, "bold"),
                        fg="white", bg="#5a5f78", activebackground="#3b3e54", padx=15, pady=5, bd=0, relief="flat")
back_button.place_forget()

# Video Display Area
video_frame = Frame(root, bg="black", width=600, height=400)
video_frame.pack(side="right", fill="both", expand=True)

video_label = Label(video_frame, bg="black")
video_label.pack(fill="both", expand=True)

# Gesture Output Label
gesture_label = Label(video_frame, text="Detected Gesture: None", font=("Arial", 14, "bold"), fg="white", bg="black")
gesture_label.pack(pady=10)

# Description on Right Side
description_label = Label(video_frame, text="🔹 Welcome to SignSpeak! 🔹\n\n"
                                            "SignSpeak is an AI-powered application that converts real-time "
                                            "hand gestures into speech, making communication easier for people "
                                            "who rely on sign language.\n\n"
                                            "✨ How It Works:\n"
                                            "Click 'Start Camera' to begin.\n"
                                            "The AI will detect your hand gestures using computer vision.\n"
                                            "Recognized gestures will be translated into speech output.\n\n"
                                            "💡 Ideal for individuals with hearing or speech impairments, "
                                            "SignSpeak bridges communication gaps and enhances accessibility.\n\n"
                                            "Start now and experience the future of AI-driven sign language interpretation! 🌍✨", font=("Arial", 12, "bold"),
                                            fg="white", wraplength=400, justify="center", background="#5a5f78", padx=200, pady=50)
description_label.pack(pady=20)

root.mainloop()
