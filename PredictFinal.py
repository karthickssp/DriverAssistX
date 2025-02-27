from cvzone.HandTrackingModule import HandDetector
from tensorflow.keras.models import load_model  # type: ignore
from tkinter import Label, Button, filedialog, Frame
from PIL import Image, ImageTk
import tensorflow as tf
import tkinter as tk
import absl.logging
import numpy as np
import pyttsx3
import math
import cv2
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
absl.logging.set_verbosity(absl.logging.ERROR)


# Initialize text-to-speech engine
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)
engine.setProperty('rate', 140) 
engine.setProperty('volume', 1)

def speak_text(text):
    engine.say(text)
    engine.runAndWait()

# Hand detector
detector = HandDetector(maxHands=1)

# Constants
imgSize = 224
offset = 20
threshold = 0.75
sign_path = r"C:\WORKSPACE\DriverAssistX\Model\model_sign.h5"
hand_path = r"C:\WORKSPACE\DriverAssistX\Model\model_hand.h5"

# Load the hand classifier model
def load_hand_classifier():
    loaded_model = load_model(hand_path)
    labels = ["Turn Left", "Turn Right", "Stop the Vehicle"]
    return loaded_model, labels

# Load the sign classifier model
def load_sign_classifier():
    loaded_model = tf.keras.models.load_model(sign_path)
    classes = {
        1: 'Speed limit (20km/h)',
        2: 'Speed limit (30km/h)',
        3: 'Speed limit (50km/h)',
        4: 'Speed limit (60km/h)',
        5: 'Speed limit (70km/h)',
        6: 'Speed limit (80km/h)',
        7: 'End of speed limit (80km/h)',
        8: 'Speed limit (100km/h)',
        9: 'Speed limit (120km/h)',
        10: 'No passing',
        11: 'No passing veh over 3.5 tons',
        12: 'Right-of-way at intersection',
        13: 'Priority road',
        14: 'Yield',
        15: 'Stop the Vehicle',
        16: 'No vehicles',
        17: 'Vehicle over 3.5 tons prohibited',
        18: 'No entry',
        19: 'General caution',
        20: 'Dangerous curve left',
        21: 'Dangerous curve right',
        22: 'Double curve',
        23: 'Bumpy road',
        24: 'Slippery road',
        25: 'Road narrows on the right',
        26: 'Road work',
        27: 'Traffic signals',
        28: 'Pedestrians',
        29: 'Children crossing',
        30: 'Bicycles crossing',
        31: 'Beware of ice/snow',
        32: 'Wild animals crossing',
        33: 'End speed + passing limits',
        34: 'Turn right ahead',
        35: 'Turn left ahead',
        36: 'Ahead only',
        37: 'Go straight or right',
        38: 'Go straight or left',
        39: 'Keep right',
        40: 'Keep left',
        41: 'Roundabout mandatory',
        42: 'End of no passing',
        43: 'End no passing vehicle over 3.5 tons'
    }
    return loaded_model, classes

def init_gui():
    global top, label, sign_image, classify_button
    # Main window settings
    top = tk.Tk()
    top.geometry('1200x800')
    top.title('AI - Based Traffic Sign Recognition and Hand Gesture Recognition System')
    top.configure(background='#1B2631') 
    # Header Frame
    header_frame = Frame(top,bg='#2E4053', pady=20)
    header_frame.pack(fill=tk.X)
    # Title
    title = Label(header_frame, 
                 text="🚦 Traffic Sign & ✋ Gesture Recognition 🤖 ", 
                 font=('Helvetica', 28, 'bold'),
                 bg='#2E4053', 
                 fg='#F4D03F')
    title.pack()
    # Subtitle
    subtitle = Label(header_frame,
                    text="Upload an image for AI-powered recognition",
                    font=('Helvetica', 14),
                    bg='#2E4053', 
                    fg='#F4D03F')
    subtitle.pack(pady=5)
    # Main Content Frame
    content_frame = Frame(top, bg="#1B2631")
    content_frame.pack(expand=True, fill=tk.BOTH, pady=20)
    # Image Display Area
    image_frame = Frame(content_frame, bg='#2E4053', padx=80, pady=80)
    image_frame.pack(expand=True)
    sign_image = Label(image_frame, bg='#2E4053',)
    sign_image.pack()
    # Status Label
    label = Label(content_frame,
                 text="📤 Upload an image to begin analysis",
                 font=('Helvetica', 16),
                 bg='#1B2631', 
                 fg='#F7F9F9')
    label.pack(pady=20)
    # Button Frame
    button_frame = Frame(top, bg='#1B2631', pady=30)
    button_frame.pack()
    # Button styles
    button_style = {
        'font': ('Helvetica', 16, 'bold'),
        'padx': 20,
        'pady': 10,
        'cursor': 'hand2',
        'relief':'raised'
    }
    # Upload Button
    upload = Button(button_frame,
                   text="📂 Upload Image",
                   command=upload_image,
                   bg='#28B463', fg='white',
                   **button_style)
    upload.pack(side=tk.LEFT, padx=20)
    # Classify Button
    classify_button = Button(button_frame,
                       text="🔍 Classify Image",
                       bg='#F39C12', 
                       fg='white', 
                       state=tk.DISABLED,
                       **button_style)
    classify_button.pack(side=tk.LEFT, padx=10)
    # Exit Button
    exit_btn = Button(button_frame,
                     text="❌ Exit",
                     command=top.destroy,
                     bg='#E74C3C',
                     fg='white',
                     **button_style)
    exit_btn.pack(side=tk.LEFT, padx=10)
    
    return top, label, sign_image, classify_button

def show_classify_button(file_path):
    classify_button.config(command=lambda: classify(file_path), state=tk.NORMAL)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        if not file_path:
            return
        uploaded = Image.open(file_path)
        uploaded.thumbnail((400, 400))  # Resize image
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='✅ Image Loaded Successfully', fg='#28B463', font=('Helvetica', 14, 'bold'))
        show_classify_button(file_path)
    except Exception as e:
        label.configure(text=f"Error: {str(e)}", fg='#f44336')
        print("Error:", e)

def classify(file_path):
    img = cv2.imread(file_path)
    hands, img = detector.findHands(img, draw=True)
    if not hands:
        print("No hand detected")
        image = cv2.imread(file_path)
        image_fromarray = Image.fromarray(image, 'RGB')
        resize_image = image_fromarray.resize((30, 30))
        expand_input = np.expand_dims(resize_image,axis=0)
        input_data = np.array(expand_input)
        input_data = input_data/255
        pred = sign_model.predict(input_data)
        result = pred.argmax()
        sign = sign_classes.get(result + 1, "Unknown sign")
        label.configure(text=f"🔍 Classified: {sign}", fg='#F39C12', font=('Helvetica', 14, 'bold'))
        top.update_idletasks()
        speak_text(sign)
        print(sign)
    else:
        print("Hand detected")
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
        prediction = hand_model.predict(imgRGB)
        classIndex = np.argmax(prediction)
        confidence = prediction[0][classIndex]
        if confidence > threshold:
            sign = hand_classes[classIndex]
        else:
            sign = "Unknown"
        label.configure(text=f"🔍 Classified: {sign}", fg='#F39C12', font=('Helvetica', 14, 'bold'))
        top.update_idletasks()
        speak_text(sign)
        print(sign)

if __name__ == "__main__":
    sign_model, sign_classes = load_sign_classifier()
    hand_model, hand_classes = load_hand_classifier()
    top, label, sign_image, classify_button = init_gui()
    top.mainloop()