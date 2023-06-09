import os
import mne
import openpyxl
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from remove_artifact import read_data, generate_data_and_label
from utils import concatenate_features

from tensorflow.keras.layers import Conv1D,BatchNormalization,LeakyReLU,MaxPool1D,GlobalAveragePooling1D,Dense,Dropout,AveragePooling1D,Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.backend import clear_session
from sklearn.model_selection import train_test_split, KFold


dir_path = "../openmiir/eeg/processed/"
print("tf: ", tf.__version__)

def cnnmodel():
    clear_session()
    model=Sequential()
    model.add(Conv1D(filters=32,kernel_size=3,strides=1,input_shape=(577,64)))#1
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPool1D(pool_size=2,strides=2))#2
    # model.add(AveragePooling1D(pool_size=2,strides=2))#6
    model.add(Conv1D(filters=16,kernel_size=3,strides=1))#3
    model.add(LeakyReLU())
    model.add(MaxPool1D(pool_size=2,strides=2))#4
    # model.add(AveragePooling1D(pool_size=2,strides=2))#6
    model.add(Dropout(0.5))
    model.add(Conv1D(filters=8,kernel_size=3,strides=1))#5
    model.add(LeakyReLU())
    model.add(AveragePooling1D(pool_size=2,strides=2))#6
    # model.add(Dropout(0.5))
    model.add(Conv1D(filters=4,kernel_size=3,strides=1))#7
    model.add(LeakyReLU())
    # model.add(AveragePooling1D(pool_size=2,strides=2))#8
    # model.add(Conv1D(filters=4,kernel_size=3,strides=1))#9
    # model.add(LeakyReLU())
    # model.add(GlobalAveragePooling1D())#10
    model.add(Flatten())
    # model.add(Dense(20,activation='relu'))#11
    model.add(Dense(12, activation='softmax'))#12
    # model.add(Dense(1, activation='sigmoid'))
    
    model.compile('adam',loss='categorical_crossentropy',metrics=['accuracy'])
    # model.compile('adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == '__main__':

    # argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true', help='include P05 subject')
    parser.add_argument('--cond', type=int, default=1, help='condition number')
    parser.add_argument('--psd', action='store_true', help='use psd data')
    parser.add_argument('--down', action='store_true', help='downsample data')
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
        data, label, group = generate_data_and_label(raw, event, condition=args.cond, psd=args.psd, down=args.down)
        data_list.append(data)
        labels.append(label)
        groups.append(group)

    data_array = np.vstack(data_list)
    label_array = np.vstack(labels)
    group_array = np.hstack(groups)
    print(data_array.shape)
    print(label_array.shape)
    print(group_array.shape)

    data_array = np.swapaxes(data_array, 1, 2)
    print(data_array.shape)

    model=cnnmodel()
    model.summary()

    tf.config.experimental_run_functions_eagerly(True)
    tf.data.experimental.enable_debug_mode()
    # Assuming your data and labels are numpy arrays
    data = data_array  # Replace ... with your data array
    labels = label_array  # Replace ... with your label array

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)

    model=cnnmodel()
    model.fit(X_train, y_train, epochs=20, batch_size=32)

    # Evaluate the model on the training data
    train_accuracy = model.evaluate(X_train, y_train, verbose=0)[1]
    print("Training Accuracy:", train_accuracy)

    # Evaluate the model on the test data
    test_accuracy = model.evaluate(X_test, y_test, verbose=0)[1]
    print("Testing Accuracy:", test_accuracy)
