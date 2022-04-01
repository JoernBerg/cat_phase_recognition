from DatasetLoader import DatasetLoader
from Preprocessor import Preprocessor
from imutils import paths
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2 as cv
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers



dataset = str(Path("C:/labeledFrames"))

phases = ["1_Incision", "2_ViscousAgentInjection", "3_Rhexis", "4_Hydrodissection", "5_Phacoemulsification",
          "6_IrrigationAndAspiration", "7_CapsulePolishing", "8_LensImplantSettingUp", "9_ViscousAgentRemoval",
          "X_TonifyingAndAntibiotics"]

print("::: [INFO] loading images... :::")
imagePaths = list(paths.list_images(dataset))
pre = Preprocessor(100, 100)
dl = DatasetLoader(preprocessors=[pre])
(data, labels) = dl.load(imagePaths, phases)
print("::: [INFO] size of features matrix: {:.1f}MB :::".format(
	data.nbytes / (1024 * 1024.0)))
le = LabelEncoder()
labels = le.fit_transform(labels)
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

class_names = ['Incision', 'ViscousAgentInjection', 'Rhexis', 'Hydrodissection', 'Phacoemulsification',
               'IrrigationAndAspiration', 'CapsulePolishing', 'LensImplantSettingUp', 'ViscousAgentRemoval', 'TonifyingAndAntibiotics']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(trainX[i])
    plt.xlabel(class_names[trainY[i]])
plt.show()

trainX = trainX / 255.0
testX = testX / 255.0

model = Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer='rmsprop',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['sparse_categorical_accuracy'])

history = model.fit(trainX, trainY, epochs=10, 
                    validation_data=(testX, testY))
class_names = ['Incision', 'ViscousAgentInjection', 'Rhexis', 'Hydrodissection', 'Phacoemulsification',
               'IrrigationAndAspiration', 'CapsulePolishing', 'LensImplantSettingUp', 'ViscousAgentRemoval', 'TonifyingAndAntibiotics']

"""
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

"""



#test_loss, test_acc = model.evaluate(testX,  testY, verbose=2)
print("::: [INFO] Saving model... :::")
model.save("classifier_conv")


