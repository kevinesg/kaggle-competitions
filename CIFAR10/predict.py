from config import CIFAR10_config as config
from keras.models import load_model
import numpy as np
import pandas as pd

# Load the test set and model
print('Loading test data and trained model...')
X_test = np.load(config.DATASET + 'X_test_array.npy')
model = load_model(config.MODEL + 'best_model.h5')

# Predict classes
y_test = model.predict(X_test)

# Convert y_test to class names
classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]
pred = []
for i in y_test.argmax(axis=1):
    pred.append(classes[i])
pred = np.array(pred)

# Save to csv in required format
df = pd.DataFrame({
    'id': np.arange(1,X_test.shape[0]+1),
    'label': pred
})
df.to_csv(config.OUTPUT + 'kaggle_predictions.csv', index=False)
print('Predictions saved!')