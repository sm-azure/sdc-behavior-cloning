import csv
import cv2
import numpy as np
import tensorflow as tf

EPOCHS = 3

lines =[]
with open('./driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for row in reader:
		lines.append(row)

with open('./driving_log2.csv') as csvfile:
	reader = csv.reader(csvfile)
	for row in reader:
		lines.append(row)

# Reverse
with open('./driving_log3.csv') as csvfile:
	reader = csv.reader(csvfile)
	for row in reader:
		lines.append(row)

#Lap 2
with open('./driving_log_lap2.csv') as csvfile:
	reader = csv.reader(csvfile)
	for row in reader:
		lines.append(row)

# Fault Correction
#with open('./driving_log_faults.csv') as csvfile:
#	reader = csv.reader(csvfile)
#	for row in reader:
#		lines.append(row)


images = []
measurements = []
correction = 0.2
current_path = 'IMG/' 

def img_add(imgBGR, measurement,flip= False):	
	# Images are in BGR, convert to RGB
	imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
	images.append(imgRGB)
	measurements.append(measurement)
	if flip:
		images.append(cv2.flip(imgRGB,1))
		measurements.append(measurement*-1.0)

for line in lines:
	center_path = line[0]
	left_path = line[1]
	right_path = line[2]
	img_center = cv2.imread(current_path + center_path.split('\\')[-1])
	img_left = cv2.imread(current_path + left_path.split('\\')[-1])
	img_right = cv2.imread(current_path + right_path.split('\\')[-1])
	measurement = float(line[3])
	steering_left = measurement + correction
	steering_right = measurement - correction
	img_add(img_center, measurement, True)
	img_add(img_left, steering_left, False)
	img_add(img_right, steering_right, False)


X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Dropout, Cropping2D

model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320, 3)))
model.add(Lambda(lambda x: tf.image.rgb_to_grayscale(x)))
model.add(Lambda(lambda x: (x-128)/255))
model.add(Conv2D(24, (5,5), strides=(2, 2), padding='valid', activation='relu'))
model.add(Conv2D(36, (5,5), strides=(2, 2), padding='valid', activation='relu'))
model.add(Conv2D(48, (5,5), strides=(2, 2), padding='valid', activation='relu'))
model.add(Conv2D(64, (3,3), strides=(1, 1), padding='valid', activation='relu'))
model.add(Conv2D(64, (3,3), strides=(1, 1), padding='valid', activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2,shuffle=True, epochs=EPOCHS)

model.save('model.h1')
