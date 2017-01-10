#import some stuff
import csv
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
import math
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D

#structures to hold the training data
centerImages = []
leftImages = []
rightImages = []
col3 = []
col4 = []
col5 = []
steeringAngle = []

letterBoxTop = 60
letterBoxBottom = 140

def preprocessImage(image):
	image = image[letterBoxTop:letterBoxBottom, 0:image.shape[1]]

	image = image.astype(np.float32)
	#Normalize image
	image = image/255.0 - 0.5

	return image


#load the training data
print("loading training data")
with open('driving_log.csv', newline='') as file:
	fileReader = csv.reader(file)
	for row in fileReader:
		centerImages.append(preprocessImage(mpimg.imread(row[0].strip())))
		leftImages.append(mpimg.imread(row[1].strip())) 
		rightImages.append(mpimg.imread(row[2].strip()))
		steeringAngle.append(float(row[6].strip()))
		#print(row)
print("training data loaded")


	

#shuffle the data
X_train = centerImages
y_train = steeringAngle
X_train, y_train = shuffle(X_train, y_train)

print("shape")
print(centerImages[0].shape[0])
print(centerImages[0].shape[1])

plt.imshow(centerImages[1])
plt.show()
print("preprocessing done")

#build up the layers

ch, row, col = 80, 320, 3  # camera format

model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(ch, row, col),
        output_shape=(ch, row, col)))
model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(Flatten())
model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(512))
model.add(Dropout(.5))
model.add(ELU())
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")


#compile and train model