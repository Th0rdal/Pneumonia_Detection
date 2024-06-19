import tensorflow

from preProcessing import imagePreProcessing
from cnnModel import configCNNModel
from denseNet121Model import configDenseNet121Model
from visualization import dataVisualization, visualizeGradCAM
import argparse

import global_var

parser = argparse.ArgumentParser(description="A programm that trains a CNN model and a pre trained denseNet121 model on pneumonia x-rays")
parser.add_argument('--detailed_models', action='store_true', help="Give a detailed summary on the models")
parser.add_argument('--retrain', action='store_true', help="Retrain models")
parser.add_argument('--gradCAM', type=str, nargs='?', const='', default=None, help="Use gradient CAM. Optionally add image path")
args = parser.parse_args()
global_var.detailedSummaryFlag = args.detailed_models
global_var.retrain = args.retrain

#print(os.listdir("input/chest_xray"))
#print("Train/PNEUMONIA: ", len(os.listdir("input/chest_xray/train/PNEUMONIA")), "Images")


if args.gradCAM == '':
    path = args.gradCAM.strip()
    model = tensorflow.keras.models.load_model(global_var.pathToCNNModel)
    visualizeGradCAM(model, "CNN", path)
    model = tensorflow.keras.models.load_model(global_var.pathToDenseNetModel)
    visualizeGradCAM(model, "DenseNet", path)
    exit(0)

dataVisualization()
imagePreProcessing()

print("")

configCNNModel("cnnModel4", epoch=20, stepsPerEpoch=100, learningRate=0.001)

print("")
configDenseNet121Model("denseNet121Model4", epoch=20, stepsPerEpoch=100, learningRate=0.001)