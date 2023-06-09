import os
import mne
import openpyxl
import argparse
import numpy as np
import matplotlib.pyplot as plt

from remove_artifact import read_data, generate_data_and_label
from utils import concatenate_features

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GroupKFold, GridSearchCV
from sklearn.metrics import accuracy_score

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

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, label_array, test_size=0.2, random_state=42)
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)


    # Define the parameter grid for GridSearchCV
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
    }

    # Define the GroupKFold cross-validation
    num_splits = 5
    group_kfold = GroupKFold(num_splits)
    group_array = np.arange(X_train.shape[0]) // (X_train.shape[0] / num_splits)

    # Initialize the SVM classifier
    svm = SVC()

    # Perform GridSearchCV using GroupKFold cross-validation
    # grid_search = GridSearchCV(svm, param_grid, cv=group_kfold)

    # Fit the model on the training data
    svm.fit(X_train, y_train)

    # Get the best model based on validation score
    # best_model = grid_search.best_estimator_
    print("Training Accuracy: ", svm.score(X_train, y_train))

    # Predict on the test data using the best model
    y_pred = svm.predict(X_test)

    # Evaluate the accuracy on the test data
    accuracy = accuracy_score(y_test, y_pred)
    print("Test Accuracy:", accuracy)