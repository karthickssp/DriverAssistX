from tensorflow.keras.models import load_model  # type: ignore
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import cv2
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import pyttsx3


# Load model
def load_hand_classifier():
    path = r"C:\WORKSPACE\DriverAssistX\HandGesture\Model\model_final.h5"
    model = load_model(path)
    classes = ["Turn Left", "Turn Right", "Stop the Vehicle"]
    return model, classes


# Hand detector
detector = HandDetector(maxHands=1)

# Constants
imgSize = 224
offset = 20
threshold = 0.75

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 125)
engine.setProperty('volume', 1)


def speak_text(text):
    engine.say(text)
    engine.runAndWait()


# GUI Setup
def init_gui():
    global top, label, sign_image  # Declare as global
    top = tk.Tk()
    top.geometry('800x600')
    top.title('AI - Based Traffic Sign Recognition and Hand Gesture Recognition System')
    top.configure(background='#CDCDCD')

    label = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
    sign_image = Label(top)
    heading = Label(top, text="Predict Traffic Sign and Symbols", pady=20, font=('arial', 20, 'bold'))
    heading.configure(background='#CDCDCD', foreground='#364156')

    upload = Button(top, text="Upload an image", command=upload_image, padx=10, pady=5, cursor="hand2")
    upload.configure(background='#364156', foreground='white', font=('arial', 12, 'bold'))
    upload.pack(side=tk.BOTTOM, pady=50)

    sign_image.pack(side=tk.BOTTOM, expand=True)
    label.pack(side=tk.BOTTOM, expand=True)
    heading.pack()

    exit_app = Button(top, text="Close IT", command=top.destroy, padx=10, pady=5, cursor="hand2")
    exit_app.configure(background='#364156', foreground='white', font=('arial', 12, 'bold'))
    exit_app.pack(side=tk.BOTTOM)

    return top


def show_classify_button(file_path):
    global classify_b

    classify_b = Button(top, text="Classify Image",
                        command=lambda: classify(file_path), padx=10, pady=5, cursor="hand2")
    classify_b.configure(background='#364156', foreground='white', font=('arial', 12, 'bold'))
    classify_b.place(relx=0.79, rely=0.46)


def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        if not file_path:
            return  # If no file selected, exit function

        uploaded = Image.open(file_path)
        max_width, max_height = 300, 300
        uploaded.thumbnail((max_width, max_height))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')

        show_classify_button(file_path)  # Pass only file_path, `model` and `classes` are global
    except Exception as e:
        print("Error:", e)


def classify(file_path):
    try:
        img = cv2.imread(file_path)
        hands, img = detector.findHands(img, draw=True)
        if not hands:
            label.config(text="Prediction: No hand detected")
            return
        hand = hands[0]
        x, y, w, h = hand['bbox']
        print(f"Bounding Box: x={x}, y={y}, w={w}, h={h}")

        y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
        x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)

        imgCrop = img[y1:y2, x1:x2]
        if imgCrop.shape[0] == 0 or imgCrop.shape[1] == 0:
            label.config(text="Prediction: No hand detected")
            return

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        imgRGB = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2RGB)
        imgRGB = cv2.resize(imgRGB, (imgSize, imgSize))
        imgRGB = imgRGB / 255.0
        imgRGB = imgRGB.reshape(1, imgSize, imgSize, 3)
        prediction = model.predict(imgRGB)
        classIndex = np.argmax(prediction)
        confidence = prediction[0][classIndex]
        if confidence > threshold:
            sign = classes[classIndex]
        else:
            sign = "Unknown"
        label.config(foreground='#011638', text=sign)
        speak_text(sign)
        print(sign)

    except Exception as e:
        print("Error in classification:", e)

if __name__ == "__main__":
    model, classes = load_hand_classifier()
    top = init_gui()
    top.mainloop()
