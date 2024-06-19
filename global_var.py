import os

detailedSummaryFlag = False
retrain = False

train_dir = "resources/input/chest_xray/train"  # folder train data
test_dir = "resources/input/chest_xray/test"  # folder test data
val_dir = "resources/input/chest_xray/val"  # folder validation data
pneumonia_dir = "resources/input/chest_xray/train/PNEUMONIA"
normal_dir = "resources/input/chest_xray/train/NORMAL"
pathToCNNModel = "resources/model/cnnModel/"
pathToDenseNetModel = "resources/model/denseNetModel/"

num_pneumonia = len(os.listdir(os.path.join(train_dir, 'PNEUMONIA')))  # counts amount of files in given path
num_normal = len(os.listdir(os.path.join(train_dir, 'NORMAL')))  # counts amount of files in given path
# calculate class weight
weight_for_0 = num_pneumonia / (num_normal + num_pneumonia)  # weight for class 0 (Pneumonie)
weight_for_1 = num_normal / (num_normal + num_pneumonia)  # weight for class 1 (normal)
class_weight = {0: weight_for_0, 1: weight_for_1}  # class weights saved for a training in a dictionary

print(f"Weight for class 0: {weight_for_0:.2f}")
print(f"Weight for class 1: {weight_for_1:.2f}")

# represent the train, validation and test sets
train = None
validation = None
test = None
