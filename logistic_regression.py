import os
import mne
import openpyxl
import argparse
import numpy as np
import matplotlib.pyplot as plt

from asrpy.asrpy import ASR
from mne_icalabel import label_components
from remove_artifact import read_data, generate_data_and_label
from utils import concatenate_features

from sklearn.model_selection import GroupKFold, train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
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

    # Assuming your features array and label array are numpy arrays
    features = features  # Replace ... with your features array
    labels = label_array  # Replace ... with your label array

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Create a group array for GroupKFold using label distribution
    group_labels, group_counts = np.unique(y_train, return_counts=True)
    group_array = np.concatenate([np.full((count,), label) for label, count in zip(group_labels, group_counts)])
    # print(group_array)
    # Define logistic regression classifier
    logreg = LogisticRegression(max_iter=1000)

    # Define parameter grid for GridSearchCV
    param_grid = {
        'C': [0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }

    # Create GroupKFold cross-validator
    group_kfold = GroupKFold(n_splits=2)

    # Perform GridSearchCV with GroupKFold cross-validation
    grid_search = GridSearchCV(logreg, param_grid, cv=group_kfold)

    # Fit the GridSearchCV on the training data
    grid_search.fit(X_train, y_train, groups=group_array)

    # Get the best model from the GridSearchCV
    best_model = grid_search.best_estimator_

    # Evaluate the best model on the test set
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    y_pred = best_model.predict(X_train)
    print("Train Accuracy: ", accuracy_score(y_train, y_pred))
    print("Test Accuracy:", accuracy)