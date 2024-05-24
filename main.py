import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras


#print(os.listdir("input/chest_xray"))
#print("Train/PNEUMONIA: ", len(os.listdir("input/chest_xray/train/PNEUMONIA")), "Images")

train_dir = "input/chest_xray/train" #directory der trainingsdaten
test_dir = "input/chest_xray/test" #directory der testdaten
val_dir = "input/chest_xray/val" #directory der validationdaten

#Anzahl der Daten in Konsole ausgeben
print("========================================\nTrain set:")
num_pneumonia = len(os.listdir(os.path.join(train_dir, 'PNEUMONIA'))) #zählt die anzahl der dateien im gegebenen pfad
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

#---------------- Data Visualization: PNEUMONIA -----------------

pneumonia_dir = "input/chest_xray/train/PNEUMONIA"
pneumonia_files = [file for file in os.listdir(pneumonia_dir) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
# erstellt eine Liste von Filenamen aus dem pneumonia_dir die mit .jpg, .png oder .jpeg enden.

plt.figure(figsize=(20, 10))
for i in range(min(9, len(pneumonia_files))):  # Limit the loop to the number of image files or 9, whichever is smaller
    img_path = os.path.join(pneumonia_dir, pneumonia_files[i])
    img = plt.imread(img_path)
    plt.subplot(3, 3, i + 1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')

plt.tight_layout()
plt.show()

#---------------- Data Visualization: NORMAL -----------------

normal_dir = "input/chest_xray/train/NORMAL"
normal_files = [file for file in os.listdir(normal_dir) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

plt.figure(figsize=(20, 10))

for i in range(9):
    plt.subplot(3, 3, i + 1)
    img = plt.imread(os.path.join(normal_dir, normal_files[i]))
    plt.imshow(img, cmap='gray')
    plt.axis('off')

plt.tight_layout()
plt.show()

#sample image, image
pic_nr = 15
pic_path = "input/chest_xray/train/NORMAL"
normal_img = os.listdir(pic_path)[pic_nr]
sample_img = plt.imread(os.path.join(normal_dir, normal_img))
plt.imshow(sample_img, cmap='gray')
plt.colorbar()
plt.title('Raw Chest X Ray Image')

print(f"Sample Picture {pic_nr} loaded from {pic_path}")
print(f"The dimensions of the image are {sample_img.shape[0]} pixels width and {sample_img.shape[1]} pixels height, one single color channel.")
print(f"The maximum pixel value is {sample_img.max():.4f} and the minimum is {sample_img.min():.4f}")
print(f"The mean value of the pixels is {sample_img.mean():.4f} and the standard deviation is {sample_img.std():.4f}")

# Nutze histplot anstelle von distplot

plt.figure(figsize=(10, 6))
sns.histplot(sample_img.ravel(), kde=False, bins=32,
             label=f"Pixel Mean {np.mean(sample_img):.4f} & Standard Deviation {np.std(sample_img):.4f}")
plt.legend(loc='upper center')
plt.title('Distribution of Pixel Intensities in the Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('# Pixels in Image')
plt.xlim(0, 255)  # Begrenzung der x-Achse auf typische Pixelwerte
plt.show()

#---------------- Image Pre Processing -----------------
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("\n\n +++ Image Preprocessing start:")

image_generator = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    samplewise_center=True,
    samplewise_std_normalization=True
    #TODO: horizontal flip
)

train = image_generator.flow_from_directory(train_dir,
                                            batch_size=8,
                                            shuffle=True,
                                            class_mode='binary',
                                            target_size=(180, 180))

validation = image_generator.flow_from_directory(val_dir,
                                                batch_size=1,
                                                shuffle=False,
                                                class_mode='binary',
                                                target_size=(180, 180))

test = image_generator.flow_from_directory(test_dir,
                                            batch_size=1,
                                            shuffle=False,
                                            class_mode='binary',
                                            target_size=(180, 180))