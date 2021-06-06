from config import CIFAR10_config as config
import numpy as np
# Load the dataset
print('Loading datasets...')
X_train = np.load(config.DATASET + 'X_train_array.npy')
X_val = np.load(config.DATASET + 'X_val_array.npy')
y_train = np.load(config.DATASET + 'y_train_array.npy')
y_val = np.load(config.DATASET + 'y_val_array.npy')
print('Datasets are loaded!')

from resnet import ResNet
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

# MODEL
# Compile model
opt = SGD(lr=config.LEARNING_RATE)
model = ResNet.build(
    width=32, height=32, depth=3, classes=10,
    stages=(3, 4, 6), filters=(32, 64, 128, 256)
)

model.compile(
    loss='categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)

# Construct callbacks
early_stopping = EarlyStopping(patience=20, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(factor=0.5, patience=6)
callbacks = [early_stopping, lr_scheduler]

# Data augmentation
aug = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
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
import matplotlib.pyplot as plt

print('Saving model information...')
plot_model(model, show_shapes=True, to_file=config.MODEL + 'model.png')
model.save(config.MODEL + 'best_model.h5')

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