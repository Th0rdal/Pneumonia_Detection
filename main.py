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

#---------------- Data Visualization: DATASET in Zahlen -----------------

#Anzahl der Daten in Konsole ausgeben
print("========================================\nTrain set:")
num_pneumonia = len(os.listdir(os.path.join(train_dir, 'PNEUMONIA'))) #zählt die anzahl der dateien im gegebenen pfad
num_normal = len(os.listdir(os.path.join(train_dir, 'NORMAL'))) #zählt die anzahl der dateien im gegebenen pfad
print(f"PNEUMONIA={num_pneumonia}")
print(f"NORMAL={num_normal}")
print(" ")

print("========================================\nTest set:")
print(f"PNEUMONIA={len(os.listdir(os.path.join(test_dir, 'PNEUMONIA')))}") #zählt die anzahl der dateien im gegebenen pfad
print(f"NORMAL={len(os.listdir(os.path.join(test_dir, 'NORMAL')))}") #zählt die anzahl der dateien im gegebenen pfad
print(" ")

print("========================================\nValidation set:")
print(f"PNEUMONIA={len(os.listdir(os.path.join(val_dir, 'PNEUMONIA')))}") #zählt die anzahl der dateien im gegebenen pfad
print(f"NORMAL={len(os.listdir(os.path.join(val_dir, 'NORMAL')))}") #zählt die anzahl der dateien im gegebenen pfad
print(" ")

#---------------- Data Visualization: 9x9 PNEUMONIA -----------------

pneumonia_dir = "input/chest_xray/train/PNEUMONIA"
pneumonia_files = [file for file in os.listdir(pneumonia_dir) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
# erstellt eine Liste von Filenamen aus dem pneumonia_dir die mit .jpg, .png oder .jpeg enden.

plt.figure(figsize=(20, 10)) #plot mit 20x10 wird erstellt
for i in range(min(9, len(pneumonia_files))):  #maximal 9 bilder werden im Plot angezeigt
    img_path = os.path.join(pneumonia_dir, pneumonia_files[i])
    img = plt.imread(img_path)
    plt.subplot(3, 3, i + 1)
    plt.imshow(img, cmap='gray') #bilder in graustufen
    plt.axis('off') #keine achsenbeschriftung

plt.suptitle('Train/Pneumonia', fontsize=32)  # Setzt den Titel der gesamten Figur
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Passt das Layout an, um Platz für den Suptitel zu schaffen
plt.show()

#---------------- Data Visualization: 9x9 NORMAL -----------------

normal_dir = "input/chest_xray/train/NORMAL"
normal_files = [file for file in os.listdir(normal_dir) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
# erstellt eine Liste von Filenamen aus dem normal_dir die mit .jpg, .png oder .jpeg enden.

plt.figure(figsize=(20, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    img = plt.imread(os.path.join(normal_dir, normal_files[i]))
    plt.imshow(img, cmap='gray')
    plt.axis('off')

plt.suptitle('Train/Normal', fontsize=32)  # Setzt den Titel der gesamten Figur
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Passt das Layout an, um Platz für den Suptitel zu schaffen
plt.show()

#---------------- Data Visualization: SAMPLE IMAGE -----------------

#sample image: Hier wird ein Bild als Sample angezeigt
pic_nr = 15 #Datei die als Sample Image verwendet werden soll
pic_path = "input/chest_xray/train/NORMAL"
normal_img = os.listdir(pic_path)[pic_nr]
sample_img = plt.imread(os.path.join(normal_dir, normal_img))
plt.imshow(sample_img, cmap='gray') #image in Graustufen anzeigen
plt.colorbar()
plt.title('Raw Chest X Ray Image') #Titel des Plots

print(f"Sample Picture {pic_nr} loaded from {pic_path}")
print(f"The dimensions of the image are {sample_img.shape[0]} pixels width and {sample_img.shape[1]} pixels height, one single color channel.")
print(f"The maximum pixel value is {sample_img.max():.4f} and the minimum is {sample_img.min():.4f}")
print(f"The mean value of the pixels is {sample_img.mean():.4f} and the standard deviation is {sample_img.std():.4f}")

#---------------- Data Visualization: SAMPLE IMAGE HISTOGRAMM -----------------

#Hier wird die Verteilung der Pixelintensities (Graustufen) in einem Histogramm angezeigt
# Nutze histplot anstelle von distplot
plt.figure(figsize=(10, 6)) #Plot ist 10x6 Zoll groß
sns.histplot(sample_img.ravel(), kde=False, bins=32, #bins=32 => Anzahl der Bins auf x-Achse
             label=f"Pixel Mean {np.mean(sample_img):.4f} & Standard Deviation {np.std(sample_img):.4f}") #label mit Pixel Mean & Std. Deviation
plt.legend(loc='upper center') #legende (label mit Pixel Mean & Std. Deviation) wird in upper center des Bildes angezeigt.
plt.title('Distribution of Pixel Intensities in the Image') #title
plt.xlabel('Pixel Intensity') #label der x-Achse
plt.ylabel('# Pixels in Image') #label der y-Achse
plt.xlim(0, 255)  #begrenzung der x-Werte auf 0 - 255 (Graustufen)
plt.show() #plot anzeigen

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