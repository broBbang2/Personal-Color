import tensorflow as tf
import cv2
from keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import os
from keras.models import load_model
import time
import requests
import tkinter as tk
from tkinter import messagebox

def decode_prediction(prediction):
    color_labels = ['Spring Warm', 'Summer Cool', 'Autumn Warm', 'Winter Cool']
    predicted_index = np.argmax(prediction)
    confidence = prediction[0][predicted_index]
    return color_labels[predicted_index], confidence

def send_prediction(predicted_color, confidence):
    try:
        url = "http://your-server-address/prediction"
        data = {"color": predicted_color, "confidence": confidence}
        response = requests.post(url, json=data)
        if response.status_code != 200:
            print(f"Prediction transmission failed: {response.status_code}")
    except Exception as e:
        print(f"Error during prediction transmission: {e}")

def show_reservation_message(predicted_color):
    root = tk.Tk()
    root.attributes('-topmost', True)
    root.withdraw()
    messagebox.showinfo("Personal Color Result", f"Your predicted color: {predicted_color}")
    root.destroy()

def capture_and_predict(model):
    if model is None:
        print("Model not provided.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Unable to open the camera.")
        return

    cascade_path = os.path.expanduser("haarcascade_frontalface_default.xml")
    if not os.path.exists(cascade_path):
        print(f"File not found: {cascade_path}")
        return

    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print("Failed to load face detection file.")
        return

    start_time = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 1:
            if start_time is None:
                start_time = time.time()
            elapsed_time = time.time() - start_time

            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face_pil = Image.fromarray(face_rgb).resize((64, 64))
                face_array = np.expand_dims(img_to_array(face_pil) / 255.0, axis=0)

                prediction = model.predict(face_array)
                predicted_color, confidence = decode_prediction(prediction)

                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, f"{predicted_color} ({confidence * 100:.2f}%)", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                if elapsed_time >= 60:
                    send_prediction(predicted_color, confidence)
                    show_reservation_message(predicted_color)
                    cap.release()
                    cv2.destroyAllWindows()
                    return
        else:
            start_time = None

        cv2.imshow('Personal Color Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Load the model and run detection
model_path = "personal_color_model.h5"
model = load_model(model_path)
capture_and_predict(model)