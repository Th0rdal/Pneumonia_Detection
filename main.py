from preProcessing import imagePreProcessing
from cnnModel import configCNNModel
from denseNet121Model import configDenseNet121Model
from visualization import dataVisualization
import argparse

import global_var

parser = argparse.ArgumentParser(description="A programm that trains a CNN model and a pre trained denseNet121 model on pneumonia x-rays")
parser.add_argument('--detailed_models', action='store_true', help="Give detailed summary on the models")
args = parser.parse_args()
global_var.detailedSummaryFlag = args.detailed_models

#print(os.listdir("input/chest_xray"))
#print("Train/PNEUMONIA: ", len(os.listdir("input/chest_xray/train/PNEUMONIA")), "Images")

dataVisualization()
imagePreProcessing()

configCNNModel()
configDenseNet121Model()
