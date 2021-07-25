# CIFAR10
#
First run `data_preprocessing.py` to preprocess the raw images into .npy files.
Then run `train_model.py` to train the chosen CNN model which is a "mini ResNet".
Finally `predict.py` predicts the test set labels and saves the output inside the output folder, which is ready to be submitted to Kaggle.
This specific model `model/best_model.h5` gives an accuracy of 91.25% on Kaggle's test set.
