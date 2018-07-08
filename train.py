import csv
import cv2
import numpy as np
import tensorflow as tf
import sys

EPOCHS = 3
BATCH_SIZE = 64

lines = []
correction = 0.2
current_path = 'IMG/'

# Load all the training data


def load_train_data(file_path):
    with open(file_path) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            lines.append(row)


# Load all the images
def process_line(line, is_test = False):
    images = []
    measurements = []
    center_path = line[0]
    left_path = line[1]
    right_path = line[2]
    img_center = cv2.cvtColor(cv2.imread(
        current_path + center_path.split('\\')[-1]), cv2.COLOR_BGR2RGB)
    img_left = cv2.cvtColor(cv2.imread(
        current_path + left_path.split('\\')[-1]), cv2.COLOR_BGR2RGB)
    img_right = cv2.cvtColor(cv2.imread(
        current_path + right_path.split('\\')[-1]), cv2.COLOR_BGR2RGB)
    measurement = float(line[3])
    steering_left = measurement + correction
    steering_right = measurement - correction
    # Add all the images
    images.append(img_center)
    measurements.append(measurement)
    if(not is_test): # append only for non test data
        images.append(cv2.flip(img_center, 1))  # flipped center image 
        measurements.append(measurement*-1.0)
    images.append(img_left)
    measurements.append(steering_left)
    images.append(img_right)
    measurements.append(steering_right)
    return images, measurements

from  sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Data generator - understanding on the basis of
# https://github.com/jeremy-shannon/CarND-Behavioral-Cloning-Project/blob/master/model.py
def generate_data(lines, batch_size=128, is_test = False):
    images = []
    measurements = []
    # Shuffle the lines
    lines = shuffle(lines)
	#Run continuously	
    while True:    
        for line in lines:
            imgs, st_angles = process_line(line, is_test)
            images.extend(imgs)
            measurements.extend(st_angles)
            if(len(measurements) >= batch_size):
                yield (np.array(images), np.array(measurements))
                images = []
                measurements = []


# Load all the training run pointers
load_train_data('./driving_log.csv')
load_train_data('./driving_log2.csv')
load_train_data('./driving_log3.csv')
load_train_data('./driving_log_lap2.csv')

#Split the train and test data
train_lines, test_lines = train_test_split(lines, test_size = 0.05)

num_train_lines = len(train_lines)
num_test_lines = len(test_lines)

# Create the generators
# https://github.com/keras-team/keras/issues/5862
train_gen = generate_data(train_lines, batch_size = BATCH_SIZE, is_test = False)
valid_gen = generate_data(train_lines, batch_size = BATCH_SIZE, is_test = False)
test_gen = generate_data(test_lines, batch_size = BATCH_SIZE, is_test = True)

#X_train = np.array(images)
#y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Dropout, Cropping2D

model = Sequential()
model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: tf.image.rgb_to_grayscale(x)))
model.add(Lambda(lambda x: (x-128)/255))
model.add(Conv2D(24, (5, 5), strides=(2, 2),
                 padding='valid', activation='elu'))
model.add(Conv2D(36, (5, 5), strides=(2, 2),
                 padding='valid', activation='elu'))
model.add(Conv2D(48, (5, 5), strides=(2, 2),
                 padding='valid', activation='elu'))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), strides=(1, 1),
                 padding='valid', activation='elu'))
model.add(Conv2D(64, (3, 3), strides=(1, 1),
                 padding='valid', activation='elu'))
model.add(Flatten())
model.add(Dropout(0.2))

model.add(Dense(100, activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=EPOCHS)
model.fit_generator(generator=train_gen, steps_per_epoch=(num_train_lines*4)/BATCH_SIZE, epochs=EPOCHS,
validation_data=valid_gen, validation_steps=(num_train_lines*4)*0.2/BATCH_SIZE)

print(model.summary())
model.save('model.h1')
