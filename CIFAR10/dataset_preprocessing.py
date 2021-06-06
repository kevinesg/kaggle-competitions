from config import CIFAR10_config as config
import os
import numpy as np
import pandas as pd
import cv2
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# PREPROCESSING
# Rename training images
path = 'dataset/train/'
for filename in os.listdir(path):
    add_zeroes = '0' * (12 - len(filename))
    new_filename = add_zeroes + filename
    os.rename(os.path.join(path, filename), os.path.join(path, new_filename))
    print(f'Processing {filename}...')

# Rename test images
path = 'dataset/test/'
for filename in os.listdir(path):
    add_zeroes = '0' * (12 - len(filename))
    new_filename = add_zeroes + filename
    os.rename(os.path.join(path, filename), os.path.join(path, new_filename))
    print(f'Processing {filename}...')

print('Done renaming!')

# Get the list of image paths
train_paths = list(paths.list_images(config.DATASET + 'train/'))
test_paths = list(paths.list_images(config.DATASET + 'test/'))
# Initialize full dataset images
X_train_full = []
X_test = []
# Read each image then convert to numpy arrays
print('Converting training images to arrays...')
for image_path in train_paths:
    print(f'Converting {image_path}...')
    image = cv2.imread(image_path)
    image_array = np.array(image).astype('float32') / 255
    X_train_full.append(image_array)
X_train_full = np.array(X_train_full)
print('Converting testing images to arrays...')
for image_path in test_paths:
    print(f'Converting {image_path}...')
    image = cv2.imread(image_path)
    image_array = np.array(image).astype('float32') / 255
    X_test.append(image_array)
X_test = np.array(X_test)
print('Done converting images!')

# Read training dataset
y_train_full = np.array(pd.read_csv('dataset/trainLabels.csv')['label'])
# One-hot encode the labels
lb = LabelBinarizer()
y_train_full = lb.fit_transform(y_train_full)

# Split full training dataset into train/val sets
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.2,
    random_state=42
)

# Save the dataset as numpy arrays
print('Saving datasets as numpy arrays...')
np.save(config.DATASET + 'X_train_array', X_train)
np.save(config.DATASET + 'X_val_array', X_val)
np.save(config.DATASET + 'y_train_array', y_train)
np.save(config.DATASET + 'y_val_array', y_val)

np.save(config.DATASET + 'X_test_array', X_test)
print('Done saving!')