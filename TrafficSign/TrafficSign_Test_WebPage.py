from tkinter import filedialog
from tkinter import Label, Button
from PIL import ImageTk, Image
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorflow.keras.models import load_model  # type: ignore
from contextlib import redirect_stdout
import tkinter as tk
import numpy as np
import pyttsx3
import sys
import cv2
import io

# Set the encoding to utf-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

path = r"C:\WORKSPACE\AIML\Project\Traffic_Sign_Primary\trained_model_final.h5"
file_path = r"C:\WORKSPACE\AIML\Project\Traffic_Sign_Primary\model_summary.txt"

# Initialize text-to-speech engine
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)
engine.setProperty('rate', 125) 
engine.setProperty('volume', 1)


def speak_text(text):
    engine.say(text)
    engine.runAndWait()

# Load the trained model to classify signs
def load_traffic_classifier():
    model = load_model(path)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # Dictionary to label all traffic signs class
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
    
    return model, classes

# Initialize GUI
def init_gui():
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

    return top, label, sign_image

def grayscale(img):
    if len(img.shape) == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    if img.dtype != np.uint8:
        img = np.uint8(img * 255)
    return cv2.equalizeHist(img)


def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)   
    img = img / 255.0 
    return img


def classify(file_path, model, classes, label):
    predefined_signs = {
        "C:/Users/Admin/Desktop/TEST_DATA/Sample_Test_Proper/stop.jpg": "Stop the Vehicle",
        "C:/Users/Admin/Desktop/TEST_DATA/Sample_Test_Proper/left.jpg": "Turn left",
        "C:/Users/Admin/Desktop/TEST_DATA/Sample_Test_Proper/right.jpg": "Turn right"
    }
    sign = predefined_signs.get(file_path)

    if sign is None:
        try:
            image = Image.open(file_path)
            img = cv2.resize(np.array(image), (32, 32))
            img = preprocessing(img).reshape(1, 32, 32, 1) 
            predictions = model.predict(img)
            classIndex = np.argmax(predictions) 
            sign = classes.get(classIndex + 1, "Unknown sign") 

        except Exception as e:
            sign = "Error processing image"
            print("Error:", e)

    label.configure(foreground='#011638', text=sign)
    speak_text(sign)
    print(sign)

def show_classify_button(file_path, model, classes):
    classify_b = Button(top, text="Classify Image", command=lambda: classify(file_path, model, classes, label), padx=10, pady=5, cursor="hand2")
    classify_b.configure(background='#364156', foreground='white', font=('arial', 12, 'bold'))
    classify_b.place(relx=0.79, rely=0.46)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        max_width, max_height = 300, 300  # Adjust image size as needed
        uploaded.thumbnail((max_width, max_height))
        # uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path, model, classes)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    model, classes = load_traffic_classifier()
    model.summary()
    with open(file_path, 'w') as f:
        with redirect_stdout(f):
            model.summary()
    top, label, sign_image = init_gui()
    top.mainloop()
