from tensorflow.keras.models import load_model # type: ignore
from cvzone.HandTrackingModule import HandDetector
from cProfile import label
import numpy as np
import warnings
import math
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='keras')

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

path = r"C:\WORKSPACE\DriverAssistX\HandGesture\Model\model_final.h5"
model = load_model(path)
print(model.input_shape)
frameWidth = 640    
frameHeight = 480
brightness = 180
threshold = 0.75       
font = cv2.FONT_HERSHEY_SIMPLEX
offset = 20
imgSize = 224

labels = ["Turn Left", "Turn Right", "Stop the Vehicle"]

while True:
    success, img = cap.read()
    if not success:
        print("Error: Could not read from webcam!")
        continue

    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]

        x, y, w, h = hand['bbox']

        y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
        x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)

        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.shape[0] == 0 or imgCrop.shape[1] == 0:
            print("Warning: Cropped image is empty, skipping frame.")
            continue

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
        imgRGB = imgRGB.reshape(1, imgSize, imgSize, 3)  # Ensure correct shape
        prediction = model.predict(imgRGB)
        classIndex = np.argmax(prediction)
        confidence = prediction[0][classIndex]
        if confidence > threshold:
            label = labels[classIndex]
        else:
            label = "Unknown"
        print(label)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        text_x, text_y = x1, y2 + 30  
        cv2.putText(img, label, (text_x, text_y), font, 1.5, (0, 255, 255), 3)

    cv2.imshow("Webcam Feed", img)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()