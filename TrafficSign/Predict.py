from re import L
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import Label, Button
from PIL import Image, ImageTk
import cv2
import tensorflow as tf


def load_traffic_classifier():

    model_path = r"C:\WORKSPACE\AIML\Project\Docker_fail\Traffic.h5"
    loaded_model = tf.keras.models.load_model(model_path)
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

def show_classify_button(file_path, model,classes):
    classify_b = Button(top, text="Classify Image", command=lambda: classify(file_path, model,classes, label), padx=10, pady=5, cursor="hand2")
    classify_b.configure(background='#364156', foreground='white', font=('arial', 12, 'bold'))
    classify_b.place(relx=0.79, rely=0.46)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        max_width, max_height = 300, 300
        uploaded.thumbnail((max_width, max_height))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path, model, classes)
    except Exception as e:
        print("Error:", e)

def classify(file_path, model,classes, label):
    imagename = file_path
    image = cv2.imread(imagename)
    image_fromarray = Image.fromarray(image, 'RGB')
    resize_image = image_fromarray.resize((30, 30))
    expand_input = np.expand_dims(resize_image,axis=0)
    input_data = np.array(expand_input)
    input_data = input_data/255
    pred = model.predict(input_data)
    result = pred.argmax()
    sign = classes.get(result + 1, "Unknown sign")
    label.config(foreground='#011638',text=sign)

if __name__ == "__main__":
    model, classes = load_traffic_classifier()
    top, label, sign_image = init_gui()
    top.mainloop()