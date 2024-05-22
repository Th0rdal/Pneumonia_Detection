import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras


#print(os.listdir("input/chest_xray"))
#print("Train/PNEUMONIA: ", len(os.listdir("input/chest_xray/train/PNEUMONIA")), "Images")

train_dir = "input/chest_xray/train"
test_dir = "input/chest_xray/test"
val_dir = "input/chest_xray/val"

#Anzahl der Daten in Konsole ausgeben
print("========================================\nTrain set:")
num_pneumonia = len(os.listdir(os.path.join(train_dir, 'PNEUMONIA')))
num_normal = len(os.listdir(os.path.join(train_dir, 'NORMAL')))
print(f"PNEUMONIA={num_pneumonia}")
print(f"NORMAL={num_normal}")
print(" ")

print("========================================\nTest set:")
print(f"PNEUMONIA={len(os.listdir(os.path.join(test_dir, 'PNEUMONIA')))}")
print(f"NORMAL={len(os.listdir(os.path.join(test_dir, 'NORMAL')))}")
print(" ")

print("========================================\nValidation set:")
print(f"PNEUMONIA={len(os.listdir(os.path.join(val_dir, 'PNEUMONIA')))}")
print(f"NORMAL={len(os.listdir(os.path.join(val_dir, 'NORMAL')))}")
print(" ")

pneumonia_dir = "input/chest_xray/train/PNEUMONIA"
pneumonia_files = [file for file in os.listdir(pneumonia_dir) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

plt.figure(figsize=(20, 10))
for i in range(min(9, len(pneumonia_files))):  # Limit the loop to the number of image files or 9, whichever is smaller
    img_path = os.path.join(pneumonia_dir, pneumonia_files[i])
    img = plt.imread(img_path)
    plt.subplot(3, 3, i + 1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')

plt.tight_layout()
plt.show()