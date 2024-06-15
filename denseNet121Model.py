import pandas as pd

from keras.layers import GlobalAveragePooling2D
from keras.models import Model
from keras.src.applications import DenseNet121
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.layers import Dense

from visualization import baseTrainingResultVisualization
import global_var


#----------transfer Learning------------
#----DenseNET----
def configDenseNet121Model():
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
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # model trained with training data, class weights and validation data used

    r = model.fit(
        global_var.train,
        epochs=10,
        validation_data=global_var.validation,
        class_weight=global_var.class_weight,
        steps_per_epoch=100,
        validation_steps=25,
    )

    evaluationDenseNet121Model(r, model)


#--------evaluation----------
def evaluationDenseNet121Model(r, model):
    baseTrainingResultVisualization(r, model)

    predicted_vals = model.predict(global_var.test, steps=len(global_var.test))

    print(confusion_matrix(global_var.test.classes, predicted_vals > 0.5))
    pd.DataFrame(classification_report(global_var.test.classes, predicted_vals > 0.5, output_dict=True))
