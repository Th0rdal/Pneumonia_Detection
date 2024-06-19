import json
import os
import shutil

import pandas as pd
import tensorflow
from keras.src.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten, BatchNormalization, Input
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.callbacks import History

from visualization import baseTrainingResultVisualization, visualizeGradCAM
import global_var


#---------------- CNN model -----------------
def configCNNModel(name="cnnModel", epoch=10, stepsPerEpoch=100, learningRate=0.001):
    print("CNN model:")

    #---------------- definition and compilation of the CNN-model -----------------
    if global_var.retrain or not os.path.exists(global_var.pathToCNNModel + name):
        # definition of the model: a sequential model with multiple layers
        model = Sequential()

        model.add(Input(shape=(180, 180, 3)))  # input layer defined
        model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))  # first convolution layer
        model.add(BatchNormalization())  # Batch-normalization
        model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))  # second convolution layer
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2, 2)))  # max pooling layer

        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))  # third convolution layer
        model.add(BatchNormalization())
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))  # forth convolution layer
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2, 2)))

        model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))  # fifth convolution layer
        model.add(BatchNormalization())
        model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))  # sixth convolution layer
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2, 2)))

        model.add(Flatten())  # convert multidimensional output of the last convolution layer to one dimensional vector
        # is needed, because dense layer in one dimensional input expected
        model.add(Dense(128, activation='relu'))  # fully connected layer with 128 neurons
        model.add(Dropout(0.2))  # dropout to prevention of overfitting (Überanpassung)

        model.add(Dense(1, activation='sigmoid'))  # output layer with one neuron and sigmoid activation function

        optimizer = Adam(learning_rate=learningRate)
        model.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

        if global_var.detailedSummaryFlag:
            model.summary()

        #---------------- model training -----------------

        # perform training
        r = model.fit(
            global_var.train,  # trainings data
            epochs=epoch,  # amount of epochs (10 full passes through trainings data)
            validation_data=global_var.validation,  # validation data
            class_weight=global_var.class_weight,  # class weights to compensate for class imbalance
            steps_per_epoch=stepsPerEpoch,  # amount of steps per epoch
            validation_steps=25,  # amount of steps for validation steps per epoch
        )

        # Clear the directory if it exists
        if os.path.exists(global_var.pathToCNNModel + name):
            shutil.rmtree(global_var.pathToCNNModel + name)
        else:
            os.makedirs(global_var.pathToCNNModel + name, exist_ok=True)

        model.save(global_var.pathToCNNModel + name)
        with open(global_var.pathToCNNModel+name+"/trainingHistory.json", "w") as f:
            json.dump(r.history, f)
    else:
        model = tensorflow.keras.models.load_model(global_var.pathToCNNModel + name)
        with open(global_var.pathToCNNModel+name+"/trainingHistory.json", "rb") as f:
            history_data = json.load(f)
        r = History()
        r.history = history_data

        if global_var.detailedSummaryFlag:
            print("\n\n +++ Model Training")
            model.summary()

    CNNModelEvaluation(r, model)

    # epoch: a full pass through the whole trainings set
    # training model with over 10 epochs with the given training and validation data
    # class weight used to compensate for imbalanced weights
    # steps_per_epoch and validation_steps set how many batch steps per epoch and validations are executed


#---------------- model evaluation with additional metrics -----------------
def CNNModelEvaluation(r, model):
    baseTrainingResultVisualization(r, model)

    # prediction of the test data
    pred = model.predict(global_var.test)

    # calculate and output confusion matrix
    print(confusion_matrix(global_var.test.classes, pred > 0.5))

    #---------------- with confusion matrix and classifications report -----------------

    # classification report created and displayed as DataFrame
    print(confusion_matrix(global_var.test.classes, pred > 0.5))
    pd.DataFrame(classification_report(global_var.test.classes, pred > 0.5, output_dict=True))

    # confusion matrix with threshold 0.7 calculated and displayed
    print(confusion_matrix(global_var.test.classes, pred > 0.7))
    # classification report with threshold of 0.7 calculated and displayed as dataframe
    pd.DataFrame(classification_report(global_var.test.classes, pred > 0.7, output_dict=True))
