from config import MNIST_config as config
from model_architecture import CNN
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


# PREPROCESSING
# Read dataset
training_data = pd.read_csv(config.DATASET + 'train.csv')
test_data = pd.read_csv(config.DATASET + 'test.csv')

# Separate features from labels
y_train_full = training_data.iloc[:, 0]
X_train_full = training_data.iloc[:, 1:]
X_test = test_data

# Rescale X to [0, 1]
X_train_full = X_train_full / 255
X_test = X_test / 255

# Reshape X into [n, 28, 28, channels]
X_train_full = np.array(X_train_full).reshape(X_train_full.shape[0], 28, 28, 1)
X_test = np.array(X_test).reshape(X_test.shape[0], 28, 28, 1)

# One-hot encode labels
lb = LabelBinarizer()
y_train_full = lb.fit_transform(y_train_full)

# Generate validation set
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.2,
    random_state=42
)


from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

# MODEL
# Compile model
opt = SGD(lr=config.LEARNING_RATE)
model = CNN.build(width=28, height=28, depth=1, classes=10)

model.compile(
    loss='categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)

# Construct callbacks
early_stopping = EarlyStopping(patience=30, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(factor=0.5, patience=10)
callbacks = [early_stopping, lr_scheduler]

# Data augmentation
aug = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False,
    fill_mode='nearest'
)

# Train model
H = model.fit(
    aug.flow(X_train, y_train, batch_size=config.BATCH_SIZE),
    validation_data=(X_val, y_val),
    steps_per_epoch=len(X_train)//config.BATCH_SIZE,
    epochs=config.EPOCHS,
    callbacks=callbacks,
    verbose=1
)
print('Training done!')

# Save model
from keras.utils import plot_model

plot_model(model, show_shapes=True, to_file=config.MODEL + 'model.png')
model.save(config.MODEL + 'best_model.h5')


import matplotlib.pyplot as plt

# PLOT
plt.style.use('ggplot')
plt.figure()
for y in ['loss', 'val_loss', 'accuracy', 'val_accuracy']:
    plt.plot(np.arange(0, len(H.history[y])), H.history[y], label=y)
plt.title('Training Loss and Accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.savefig(config.OUTPUT + 'loss_acc_plot.jpg')
plt.close()
print('Model saved!')


# PREDICTIONS
# Load best model and predict test data
y_test = model.predict(X_test)
pred = y_test.argmax(axis=1)

# Save to csv in required format
df = pd.DataFrame({
    'ImageID': np.arange(1,28001),
    'Label': pred
})
df.to_csv(config.OUTPUT + 'kaggle_predictions.csv', index=False)
print('Predictions saved!')