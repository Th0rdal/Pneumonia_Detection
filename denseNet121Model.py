import json
import os
import shutil

import pandas as pd
import tensorflow

from keras.layers import GlobalAveragePooling2D
from keras.models import Model
from keras.src.applications import DenseNet121
from keras.src.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.callbacks import History

from visualization import baseTrainingResultVisualization, visualizeGradCAM
import global_var


#----------transfer Learning------------
#----DenseNET----
def configDenseNet121Model(name="cnnModel", epoch=10, stepsPerEpoch=100, learningRate=0.001):
    print("DenseNet121 model:")

    if global_var.retrain or not os.path.exists(global_var.pathToDenseNetModel + name):
        # load pre-trained DenseNet121-model, omitting the highest layer and use weights from ImageNet
        base_model = DenseNet121(input_shape=(180, 180, 3), include_top=False, weights='imagenet', pooling='avg')

        # display architecture of basemodel
        if global_var.detailedSummaryFlag:
            base_model.summary()

        # output amount of layers in basemodel
        layers = base_model.layers
        print(f"\nThe model has {len(layers)} layers")

        # output input and output form of the basemodel
        print(f"\nThe input shape {base_model.input}")
        print(f"The output shape {base_model.output}\n")

        # create new model by adding layers to the basemodel
        # model = Sequential()
        base_model = DenseNet121(include_top=False, weights='imagenet')
        x = base_model.output
        x = GlobalAveragePooling2D()(x)  # Global Average Pooling-Schicht hinzufügen
        predictions = Dense(1, activation="sigmoid")(
            x)  # Dense-Schicht mit Sigmoid-Aktivierung für binäre Klassifikation hinzufügen

        # the full model defined
        model = Model(inputs=base_model.input, outputs=predictions)
        # model.add(base_model)
        # model.add(GlobalAveragePooling2D())
        # model.add(Dense(1, activation='sigmoid'))

        # model of binary cross entropy loss, adam optimizer and compile accuracy matrix
        optimizer = Adam(learning_rate=learningRate)
        model.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

        # model trained with training data, class weights and validation data used

        r = model.fit(
            global_var.train,
            epochs=epoch,
            validation_data=global_var.validation,
            class_weight=global_var.class_weight,
            steps_per_epoch=stepsPerEpoch,
            validation_steps=25,
        )

        # Clear the directory if it exists
        if os.path.exists(global_var.pathToDenseNetModel + name):
            shutil.rmtree(global_var.pathToDenseNetModel + name)
        else:
            os.makedirs(global_var.pathToDenseNetModel + name, exist_ok=True)

        model.save(global_var.pathToDenseNetModel + name)
        with open(global_var.pathToDenseNetModel+name+"/trainingHistory.json", "w") as f:
            json.dump(r.history, f)
    else:
        model = tensorflow.keras.models.load_model(global_var.pathToDenseNetModel + name)
        with open(global_var.pathToDenseNetModel+name+"/trainingHistory.json", "rb") as f:
            history_data = json.load(f)
        r = History()
        r.history = history_data

        if global_var.detailedSummaryFlag:
            model.summary()

    evaluationDenseNet121Model(r, model)


#--------evaluation----------
def evaluationDenseNet121Model(r, model):
    baseTrainingResultVisualization(r, model)

    predicted_vals = model.predict(global_var.test, steps=len(global_var.test))

    print(confusion_matrix(global_var.test.classes, predicted_vals > 0.5))
    pd.DataFrame(classification_report(global_var.test.classes, predicted_vals > 0.5, output_dict=True))
