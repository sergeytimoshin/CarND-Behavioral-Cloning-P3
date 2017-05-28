import os
import csv
import cv2
import numpy as np
import pandas as pd
import sklearn
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import random_shift
from tqdm import tqdm
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Convolution2D, Dropout

def filter_images(csv, grouped=False):
    df = pd.read_csv(csv, header=None)
    if grouped:
        bins = np.linspace(-1, 1, 400)
        groups = df.groupby(np.digitize(df[3],bins)).head(20)
        return groups.as_matrix()
    else:
        return df.as_matrix()
    
samples = []
csv_file = './data/driving_log.csv'
data_dir = './data/IMG/'
samples = filter_images(csv_file, grouped=False)
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def preprocess_image(image):
    image = random_brightness(image)
    image = random_shadow(image)
    image = random_shift(image, 0, 0.2, 0, 1, 2)
    # image = triangle_crop(image)
    return image

def triangle_crop(image):
    stencil = np.zeros(image.shape).astype(image.dtype)
    ysize = image.shape[0]
    xsize = image.shape[1]
    left_bottom = [0, ysize]
    right_bottom = [xsize, ysize]
    apex = [xsize/2, ysize/1.7]

    contours = [np.array([left_bottom, right_bottom, apex], dtype='int32')]
    color = [255, 255, 255]
    cv2.fillPoly(stencil, contours, color)
    result = cv2.bitwise_and(image, stencil)
    return result
    
def random_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[:,:,2] = hsv[:,:,2] * np.random.uniform(0.6,0.95)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return image

def random_shadow(image):
    image_a = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
    w, h = image.shape[0], image.shape[1]
    x0, y0 = 0, np.random.randint(0, h)
    x1, y1 = w, np.random.randint(y0, h)
    
    overlay = image_a.copy()
    cv2.line(overlay,(y0, x0),(y1, x1),(0,0,0,0),np.random.randint(10,50))
    cv2.addWeighted(overlay, 0.4, image_a, 0.6, 0, image_a)
    image = cv2.cvtColor(image_a, cv2.COLOR_RGBA2RGB)
    return image

def generator(samples, augment, batch_size):
    steering_corrections = [-0.25, 0, 0.25]
    num_samples = len(samples)
    while True:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    name = data_dir + batch_sample[i].split('/')[-1]
                    image = cv2.imread(name)
                    angle = float(batch_sample[3]) + steering_corrections[i]

                    images.append(image)
                    angles.append(angle)

            augmented_images, augmented_angles = [], []
            if augment:
                for image, angle in zip(images, angles):
                    augmented_images.append(image)
                    augmented_angles.append(angle)

                    augmented_images.append(cv2.flip(image, 1))
                    augmented_angles.append(angle * -1.0)
                    
                    augmented_images.append(preprocess_image(image))
                    augmented_angles.append(angle)

                    augmented_images.append(preprocess_image(cv2.flip(image, 1)))
                    augmented_angles.append(angle * -1.0)
            else:
                augmented_images = images
                augmented_angles = angles
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, augment=True, batch_size=128)
validation_generator = generator(validation_samples, augment=False, batch_size=128)

row, col, ch = 160, 320, 3

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(row, col, ch),
                 output_shape=(row, col, ch)))
model.add(Cropping2D(cropping=((60, 20), (0, 0))))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(32, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit_generator(generator=train_generator,
                    validation_data=validation_generator,
                    samples_per_epoch=len(train_samples)*12,
                    nb_val_samples=len(validation_samples),
                    nb_epoch=30,
                    callbacks=[ModelCheckpoint('model.h5',save_best_only=True)],
                    verbose=2)
