# DriverAssistX

## Overview
DriverAssistX is an AI-powered system designed for real-time road sign recognition and hand gesture detection to assist human drivers. It utilizes deep learning and computer vision techniques to enhance road safety and driver assistance.

## Features

- 🚦 **Traffic Sign Recognition** using a trained CNN model.
- ✋ **Hand Gesture Detection** using OpenCV and MediaPipe.
- 🎥 **Real-Time Detection** via a webcam.
- 🔄 **Seamless Switching** between sign and gesture recognition based on detected input.
- 📢 **Voice Feedback** for recognized signs and gestures.

## Technologies Used

- **Programming Languages:** Python
- **Libraries & Frameworks:** OpenCV, MediaPipe, TensorFlow/Keras, Tkinter, PyTTSX3
- **Machine Learning:** Convolutional Neural Networks (CNN)
- **Hardware:** Webcam for real-time video feed

## Installation

### Prerequisites
Ensure you have Python installed (3.8+ recommended) and install the required dependencies:
```sh
pip install opencv-python mediapipe tensorflow keras numpy pillow pyttsx3
```

### Clone the Repository
```sh
git clone https://github.com/karthickssp/DriverAssistX.git
cd DriverAssistX
```

### Run the Application
```sh
python PredictFinal.py
```

## Usage

- **Upload an Image:** Select an image for classification.
- **Real-Time Detection:** Use a webcam to detect road signs or hand gestures.
- **Voice Feedback:** Recognized inputs are displayed on the screen and spoken aloud.

## Project Structure
```
DriverAssistX/
|── HandGesture/          # Dataset and Model for gesture recognition
│── Models/               # Pretrained models for sign & gesture recognition
|── Output/               # Result and other outcomes
│── Test_Data/            # Sample images for testing
|── TrafficSign/          # Dataset and Model for Sign recognition
│── src/
│   │── PredictFinal.py   # Entry point for the application, GUI implementation using Tkinter, Sign & gesture recognition logic
│── README.md             # Project documentation
```

## Future Enhancements

-✅ Improve model accuracy with more training data.
-✅ Implement a mobile application version.
-✅ Support additional traffic rules and hand gestures.

## Contributors
Karthick ([karthickssp00@gmail.com](mailto:karthickssp00@gmail.com))


🚀 Drive safely with AI-powered assistance!
