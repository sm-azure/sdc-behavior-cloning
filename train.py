import csv
import cv2
import numpy as np

lines =[]
with open('./driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for row in reader:
		lines.append(row)

images = []
measurements = []
for line in lines:
	source_path = line[0]
	filename = source_path.split('\\')[-1]
	current_path = 'IMG/' + filename
	image = cv2.imread(current_path)
	images.append(image)
	images.append(cv2.flip(image,1))
	measurement = float(line[3])
	measurements.append(measurement)
	measurements.append(measurement*-1.0)

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Dropout

model = Sequential()
model.add(Lambda(lambda x: (x-128)/255, input_shape=(160,320,3)))
model.add(Conv2D(5, (10,10), strides=(1, 1), padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'))
model.add(Conv2D(10, (5,5), strides=(1, 1), padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'))
model.add(Flatten())
model.add(Dense(64))
model.add(Dropout(0.5))
model.add(Dense(8))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2,shuffle=True, epochs=3)

model.save('model.h1')
