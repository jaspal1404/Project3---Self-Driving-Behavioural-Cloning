
#import os
import csv
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Cropping2D, Lambda
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
#from keras.backend import tf as ktf
import cv2


# Load training image paths into a list
samples = []
#with open('./data/driving_log.csv') as csvfile:
with open('./training_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# Shuffle the data
random.shuffle(samples)


# Split data into training and validation sets
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
print(len(train_samples), len(validation_samples))
          
                
# Converts RGB images to gray scale
def Color2Gray(img_data):
    X = 0.299 * img_data[:, :, :, 0] + 0.587 * img_data[:, :, :, 1] + 0.114 * img_data[:, :, :, 2]
    X = X.reshape(X.shape + (1,))
    return X


# Generator function to process data on the fly
def generator(samples, batch_size):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                #name = './data/'+batch_sample[0]
                name = './training_data/IMG_CENTER/'+batch_sample[0].split('\\')[-1]
                center_image = cv2.imread(name)
                #center_image_resized = cv2.resize(center_image, (32, 32))
                #center_image_rgb = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X = np.array(images)
            y = np.array(angles)
            X = Color2Gray(X)           # Preprocessing images to gray scale
            yield shuffle(X, y)
            

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=64)
validation_generator = generator(validation_samples, batch_size=64)


# Model architecture
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,1)))
model.add(Cropping2D(cropping=((70,20), (0,0))))
model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Convolution2D(16, 5, 5, activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Convolution2D(32, 5, 5, activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Convolution2D(64, 5, 5, activation='relu'))
model.add(Flatten())
model.add(Dense(512))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=5)


model.save('model.h5')























