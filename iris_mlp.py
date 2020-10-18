#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import argparse


def load_data(data_file):
    """ Reads IRIS CSV file, extract features and labels, and convert labels to one-hot code
    
    data_file: IRIS CSV file path
    """
    
    # Read data CSV
    iris_data = pd.read_csv(data_file)
    
    # Get column names
    iris_data_cols = iris_data.columns
    
    # Extract features (predictors) from the first 4 columns
    predictors = iris_data[iris_data_cols[iris_data_cols != 'class']]
    
    # Extract labels (target) from the last column
    target = iris_data['class']
    # Convert the 3 labels to numbers 0,1,2
    target = target.map({"iris_setosa":0,"iris_versicolor":1, "iris_virginica":2}).values
    # Convert the labels to one hot code
    target = to_categorical(target)
    
    # Get the number of feature columns
    n_cols = predictors.shape[1]
    
    return predictors, target, n_cols


def norm(predictors, normalize=None):
    """ Normalize predictors with MEAN-STD or MIN-MAX
    
    predictors: as returned by the load_data() function
    normalize: "std" for MEAN-STD or "minmax" for MIN-MAX normalization
    """
    
    if normalize == "std":
        predictors_norm = (predictors - predictors.mean()) / predictors.std()
    elif normalize == "minmax":
        predictors_norm = (predictors - predictors.min()) / (predictors.max() - predictors.min())

    return predictors_norm


def split_data(predictors_norm, target):
    """ Split data into train and test
    
    predictors_norm: as returned by the norm() function
    target: as returned by the load_data() function
    """
    
    x_train, x_test, y_train, y_test = train_test_split(predictors_norm, target, test_size=0.05)

    return x_train, x_test, y_train, y_test


def iris_model(dimensionality, n_cols, learning_rate):
    """ Define the model

    dimensionality: size of the hidden layer
    n_cols: as returned by the load_data() function
    learning_rate: for Adam optimizer, modified by an argument
    """
    
    # Create the model
    model = Sequential()
    model.add(Dense(dimensionality, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(3, activation='softmax'))
    
    # Compile the model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model


def train_model(model, x_train, x_test, y_train, y_test, valid_ratio, epochs, batch_size):
    """ Train the model on the data, and evaluate it on the test dataset.
    
    model: as returned by the iris_model() function
    x_train, x_test, y_train, y_test: as returned by the split_data() function
    epochs, batch_size: modified by arguments
    """
    
    # Fit the model
    model.fit(x_train, y_train, validation_split=valid_ratio, epochs=epochs, batch_size=batch_size, verbose=2)
    
    print("\nEvaluating:")
    # Evaluate the model on the test data
    test_loss, test_accuracy = model.evaluate(x=x_test, y=y_test)
    
    return test_loss, test_accuracy


def predict_labels(model, x_test):
    """ Predict the labels of the test dataset

    model: after fitted/trained by the train_model() function
    x_test: as returned by the split_data() function
    """
    
    pred = model.predict(x_test)
    #pred_labels = model.predict_classes(x_test)    # depricated
    pred_labels = np.argmax(model.predict(x_test), axis=-1)
    
    return pred, pred_labels


def evaluate(y_test, pred_labels):
    """ Evaluate the model (another way)
    
    y_test: as returned by the split_data() function
    pred_labels: predictions as returned by the predict_labels() function
    """
    
    # Converts one-hot code to a label (the index of 1)
    y_test_labels = np.argmax(y_test, axis=1)
    
    # Compare test labels to predicted labels
    score = accuracy_score(y_test_labels, pred_labels)
    
    return y_test_labels, score


def main():
    argparser = argparse.ArgumentParser(description='The perceptron.')
    argparser.add_argument("--iris-data", "-d",
                           type=str, default="./data/iris_flowers.csv",
                           help="Path to the Iris flower data set")
    argparser.add_argument("--valid-ratio", "-t",
                           type=float, default=0.05,
                           help="Size of the validation set (as a ratio of the training set)")
    argparser.add_argument("--epochs", "-e",
                           type=int, default=100,
                           help="Number of epochs")
    argparser.add_argument("--batch-size", "-b",
                           type=int, default=100,
                           help="Batch size")
    argparser.add_argument("--learning-rate", "-r",
                           type=float, default=0.1,
                           help="Learning rate")
    argparser.add_argument("--normalize", "-n",
                           type=str, default="minmax",
                           help="Normalize feature values to the range [0,1] (minmax) or standardize them (std)")
    argparser.add_argument("--dimensionality", "-D",
                           type=int, default=8,
                           help="Size of the hidden layer")
    args = argparser.parse_args()


    # Train and Evaluate
    predictors, target, n_cols = load_data(args.iris_data)
    predictors_norm = norm(predictors, normalize=args.normalize)
    x_train, x_test, y_train, y_test = split_data(predictors_norm, target)

    # Build the model
    model = iris_model(args.dimensionality, n_cols, args.learning_rate)

    # Train the model
    test_loss, test_accuracy = train_model(model, x_train, x_test, y_train, y_test, args.valid_ratio, args.epochs, args.batch_size)
    print("Test Loss:", test_loss, "Test Accuracy:", test_accuracy)

    # Predict the labels of the test dataset
    pred, pred_labels = predict_labels(model, x_test)
    print("\nLabel Predictions (Probabilities):\n", pred)
    print("\nLabel Predictions (Classes):\n", pred_labels)

    # Evaluate model's predictions on the test dataset labels
    y_test_labels, score = evaluate(y_test, pred_labels)
    print("\nTest Labels vs. Predicted Labels:\n", y_test_labels, "\n", pred_labels, sep = '')
    print("\nAccuracy Score:", score, "\n")

    print("Current Arguments:", args.__dict__, "\n")


if __name__ == "__main__":
    main()
