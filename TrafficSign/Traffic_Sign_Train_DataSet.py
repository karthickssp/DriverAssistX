import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import cv2
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import seaborn as sns

path = r"C:\WORKSPACE\DriverAssistX\TrafficSign\DataSet\Dataset"  
labelFile = r"C:\WORKSPACE\DriverAssistX\TrafficSign\DataSet\labels.csv"
batch_size_val = 50
steps_per_epoch_val = 2000
epochs_val = 30
imageDimensions = (32, 32, 3)
testRatio = 0.2
validationRatio = 0.2

# Importing Images
images = []
classNo = []
myList = os.listdir(path)
print("Total Classes Detected:", len(myList))
noOfClasses = len(myList)

print("Importing Classes...")
for count, class_dir in enumerate(myList):
    myPicList = os.listdir(f"{path}/{class_dir}")
    for img_name in myPicList:
        curImg = cv2.imread(f"{path}/{class_dir}/{img_name}")
        if curImg is None:
            print(f"Image not loaded correctly: {path}/{class_dir}/{img_name}")
        else:
            curImg = cv2.resize(curImg, (imageDimensions[0], imageDimensions[1]))
            images.append(curImg)
            classNo.append(count)

print("Imported Classes.")
images = np.array(images)
classNo = np.array(classNo)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validationRatio)

# Verify Data Shapes
print("Data Shapes")
print("Train:", X_train.shape, y_train.shape)
print("Validation:", X_validation.shape, y_validation.shape)
print("Test:", X_test.shape, y_test.shape)

# Read CSV File
data = pd.read_csv(labelFile)
print("Data shape:", data.shape)

# Display Dataset Distribution
num_of_samples = [len(X_train[y_train == i]) for i in range(noOfClasses)]
plt.figure(figsize=(12, 6))
plt.bar(range(noOfClasses), num_of_samples)
plt.title("Distribution of the Training Dataset")
plt.xlabel("Class Number")
plt.ylabel("Number of Images")
plt.grid(True)
plt.savefig("dataset_distribution.png")
plt.show()

# Preprocess Images
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def equalize(img):
    return cv2.equalizeHist(img)

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255.0
    return img

X_train = np.array(list(map(preprocessing, X_train))).reshape(X_train.shape[0], imageDimensions[0], imageDimensions[1], 1)
X_validation = np.array(list(map(preprocessing, X_validation))).reshape(X_validation.shape[0], imageDimensions[0], imageDimensions[1], 1)
X_test = np.array(list(map(preprocessing, X_test))).reshape(X_test.shape[0], imageDimensions[0], imageDimensions[1], 1)

dataGen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, shear_range=0.1, rotation_range=10)
dataGen.fit(X_train)

# Convert Labels to One-Hot Encoding
y_train = to_categorical(y_train, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)

# CNN Model Definition
def myModel():
    no_Of_Filters = 60
    size_of_Filter = (5, 5)
    size_of_Filter2 = (3, 3)
    size_of_pool = (2, 2)
    no_Of_Nodes = 500

    model = Sequential()
    model.add(Conv2D(no_Of_Filters, size_of_Filter, input_shape=(imageDimensions[0], imageDimensions[1], 1), activation='relu'))
    model.add(Conv2D(no_Of_Filters, size_of_Filter, activation='relu'))
    model.add(MaxPooling2D(pool_size=size_of_pool))

    model.add(Conv2D(no_Of_Filters // 2, size_of_Filter2, activation='relu'))
    model.add(Conv2D(no_Of_Filters // 2, size_of_Filter2, activation='relu'))
    model.add(MaxPooling2D(pool_size=size_of_pool))

    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(no_Of_Nodes, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))

    model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train Model
model = myModel()
print(model.summary())

history = model.fit(dataGen.flow(X_train, y_train, batch_size=batch_size_val),
                    steps_per_epoch=steps_per_epoch_val,
                    epochs=epochs_val,
                    validation_data=(X_validation, y_validation),
                    shuffle=True)

# Save Model
model.save("trained_model.h5")

# Plot Training History
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig("training_accuracy.png")
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig("training_loss.png")
plt.show()

# Evaluate Model
score = model.evaluate(X_test, y_test, verbose=0)
print('Test Loss:', score[0])
print('Test Accuracy:', score[1])

# Save Metrics to File
with open("evaluation_metrics.txt", "w") as f:
    f.write(f"Test Loss: {score[0]}\n")
    f.write(f"Test Accuracy: {score[1]}\n")

# Generate Classification Report and Confusion Matrix
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

report = classification_report(y_true, y_pred_classes, target_names=[str(i) for i in range(noOfClasses)])
conf_matrix = confusion_matrix(y_true, y_pred_classes)

with open("classification_report.txt", "w") as f:
    f.write(report)

# Plot Confusion Matrix
plt.figure(figsize=(12, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[str(i) for i in range(noOfClasses)],
            yticklabels=[str(i) for i in range(noOfClasses)])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig("confusion_matrix.png")
plt.show()
