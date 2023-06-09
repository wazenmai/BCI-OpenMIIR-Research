import os
import mne
import openpyxl
import argparse
import numpy as np
import matplotlib.pyplot as plt

from remove_artifact import read_data, generate_data_and_label
from utils import concatenate_features

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score

dir_path = "./openmiir/eeg/processed/"

if __name__ == '__main__':

    # argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true', help='include P05 subject')
    parser.add_argument('--cond', type=int, default=1, help='condition number')
    args = parser.parse_args()

    # Read data (Exclude subject 5)
    subjects = []
    for root, dir, files in os.walk(dir_path):
        for f in files:
            if not args.all and "05" in f:
                continue
            if "-rec.fif" in f:
                subjects.append(f)
    print(subjects)

    data_list, labels, groups = [], [], []
    for subject in subjects:
        raw, event = read_data(dir_path, subject, filter=False, reconstructed=True)
        data, label, group = generate_data_and_label(raw, event, condition=args.cond)
        data_list.append(data)
        labels.append(label)
        groups.append(group)

    data_array = np.vstack(data_list)
    label_array = np.hstack(labels)
    group_array = np.hstack(groups)
    print(data_array.shape)
    print(label_array.shape)
    print(group_array.shape)

    features=[]
    for data in data_array:
        features.append(concatenate_features(data))
    features=np.array(features)
    features.shape

    data = features
    labels = label_array

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Initialize the KNN classifier
    knn = KNeighborsClassifier()

    # Fit the model on the training data
    knn.fit(X_train, y_train)
    print("Training Accuracy:", knn.score(X_train, y_train))

    # Predict on the test data
    y_pred = knn.predict(X_test)

    # Evaluate the accuracy on the test data
    accuracy = np.mean(y_pred == y_test)
    print("Test Accuracy:", accuracy)

    # Perform cross-validation
    num_folds = 5  # Change this according to the desired number of cross-validation folds
    scores = cross_val_score(knn, data, labels, cv=num_folds)

    # Print the cross-validation scores
    print("Cross-Validation Scores:", scores)
    print("Mean Cross-Validation Score:", np.mean(scores))
