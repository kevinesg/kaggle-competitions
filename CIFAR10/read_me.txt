Dataset cannot be uploaded because the filesize is too large.
Download the zipped dataset from Kaggle then unzip it as is. The code is formmated in such a way that it is ready to handle the unzipped files.

First run "data_preprocessing.py" to preprocess the raw images into .npy files.
Then run "train_model.py" to train the chosen CNN model which is mini ResNet.
Finally "predict.py" predicts the test set labels and saves the output inside the output folder, which is ready to be submitted to Kaggle.
This specific model "model/best_model.h5" gives an accuracy of 91.25% on Kaggle's test set.
