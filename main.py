import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras

#print(os.listdir("input/chest_xray"))
#print("Train/PNEUMONIA: ", len(os.listdir("input/chest_xray/train/PNEUMONIA")), "Images")

train_dir = "input/chest_xray/train"  #folder train data
test_dir = "input/chest_xray/test"  #folder test data
val_dir = "input/chest_xray/val"  #folder validation data

#---------------- Data Visualization: DATASET in numbers -----------------

# print amount of data in console
print("========================================\nTrain set:")
num_pneumonia = len(os.listdir(os.path.join(train_dir, 'PNEUMONIA')))  #counts amount of files in given path
num_normal = len(os.listdir(os.path.join(train_dir, 'NORMAL')))  #counts amount of files in given path
print(f"PNEUMONIA={num_pneumonia}")
print(f"NORMAL={num_normal}")
print(" ")

print("========================================\nTest set:")
print(f"PNEUMONIA={len(os.listdir(os.path.join(test_dir, 'PNEUMONIA')))}")  #counts amount of files in given path
print(f"NORMAL={len(os.listdir(os.path.join(test_dir, 'NORMAL')))}")  #counts amount of files in given path
print(" ")

print("========================================\nValidation set:")
print(f"PNEUMONIA={len(os.listdir(os.path.join(val_dir, 'PNEUMONIA')))}")  #counts amount of files in given path
print(f"NORMAL={len(os.listdir(os.path.join(val_dir, 'NORMAL')))}")  #counts amount of files in given path
print(" ")

#---------------- Data Visualization: 9x9 PNEUMONIA -----------------

pneumonia_dir = "input/chest_xray/train/PNEUMONIA"
pneumonia_files = [file for file in os.listdir(pneumonia_dir) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
# erstellt eine Liste von Filenamen aus dem pneumonia_dir die mit .jpg, .png oder .jpeg enden.

plt.figure(figsize=(20, 10))  # create 20x10 plot
for i in range(min(9, len(pneumonia_files))):  #max 9 pictures are shown in plot
    img_path = os.path.join(pneumonia_dir, pneumonia_files[i])
    img = plt.imread(img_path)
    plt.subplot(3, 3, i + 1)
    plt.imshow(img, cmap='gray')  #pictures in greyscale
    plt.axis('off')  # no axis label

plt.suptitle('Train/Pneumonia', fontsize=32)  # sets title of figure
plt.tight_layout(rect=[0, 0, 1, 0.95])  # fits layout to make space for subtitle
plt.show()

#---------------- Data Visualization: 9x9 NORMAL -----------------

normal_dir = "input/chest_xray/train/NORMAL"
normal_files = [file for file in os.listdir(normal_dir) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
# creates list of filenames from normal_dir with .jpg, .png or .jpeg ending

plt.figure(figsize=(20, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    img = plt.imread(os.path.join(normal_dir, normal_files[i]))
    plt.imshow(img, cmap='gray')
    plt.axis('off')

plt.suptitle('Train/Normal', fontsize=32)  # sets title of figure
plt.tight_layout(rect=[0, 0, 1, 0.95])  # fits layout to make space for subtitle
plt.show()

#---------------- Data Visualization: SAMPLE IMAGE ----------------

# sampe image: shows picture as sample
pic_nr = 15  # file that should be used as sample image
pic_path = "input/chest_xray/train/NORMAL"
normal_img = os.listdir(pic_path)[pic_nr]
sample_img = plt.imread(os.path.join(normal_dir, normal_img))
plt.imshow(sample_img, cmap='gray')  # show image in greyscale
plt.colorbar()
plt.title('Raw Chest X Ray Image')  # title of the Plots

print(f"Sample Picture {pic_nr} loaded from {pic_path}")
print(
    f"The dimensions of the image are {sample_img.shape[0]} pixels width and {sample_img.shape[1]} pixels height, one single color channel.")
print(f"The maximum pixel value is {sample_img.max():.4f} and the minimum is {sample_img.min():.4f}")
print(f"The mean value of the pixels is {sample_img.mean():.4f} and the standard deviation is {sample_img.std():.4f}")

#---------------- Data Visualization: SAMPLE IMAGE HISTOGRAMM -----------------

# shows distribution of the pixel intensity in a histogram
# uses histplot instead of distplot
plt.figure(figsize=(10, 6))  # plot size 10x6
sns.histplot(sample_img.ravel(), kde=False, bins=32,  # bins=32 => amount of bins in x-axis
             label=f"Pixel Mean {np.mean(sample_img):.4f} & Standard Deviation {np.std(sample_img):.4f}")  # label pixel mean & Std. Deviation
plt.legend(
    loc='upper center')  # legend (label mit Pixel Mean & Std. Deviation) displayed in upper center of the picture
plt.title('Distribution of Pixel Intensities in the Image')  # title
plt.xlabel('Pixel Intensity')  # x-axis label
plt.ylabel('# Pixels in Image')  # y-axis label
plt.xlim(0, 255)  # limits the x values between 0 and 255 (greyscale)
plt.show()  # show plot

#---------------- Image Pre Processing -----------------
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("\n\n +++ Image Preprocessing start:")

# Create an instance of ImageDataGenerator with specified augmentation parameters
image_generator = ImageDataGenerator(
    rotation_range=20,  # Rotate the image randomly up to 20 degrees
    width_shift_range=0.1,  # Shift the image horizontally by up to 10% of the width
    shear_range=0.1,  # Shear the image by up to 10%
    zoom_range=0.1,  # Zoom in/out on the image by up to 10%
    samplewise_center=True,  # Center the image data
    samplewise_std_normalization=True)  # Normalize the image data to have standard deviation 1

# Generate batches of tensor image data with real-time data augmentation for training set
train = image_generator.flow_from_directory(train_dir,
                                            batch_size=8,  # Number of images to process at a time / # Images per batch
                                            shuffle=True,
                                            class_mode='binary',
                                            target_size=(180, 180))  # Resize the images to 180x180 pixels

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

# check the pixel value of the presentation
generated_image, label = train.__getitem__(0)
print("\n")
print(f'Pixelwerte vor der Normalisierung: {generated_image[0].min()}, {generated_image[0].max()}')

# ensure that the picture data are between [0, 1]
normalized_image = (generated_image[0] - np.min(generated_image[0])) / (
            np.max(generated_image[0]) - np.min(generated_image[0]))
print(f'Pixelwerte nach der Normalisierung: {normalized_image.min()}, {normalized_image.max()}')
print("\n")

# plot the picture
sns.set_style('white')
plt.imshow(normalized_image.astype('float32'), cmap='gray', vmin=0, vmax=1)
plt.colorbar()
plt.title('Raw Chest X Ray Image - Preprocessed')
plt.show()

print(
    f"The dimensions of the image are {generated_image.shape[1]} pixels width and {generated_image.shape[2]} pixels height, one single color channel.")
print(f"The maximum pixel value is {generated_image.max():.4f} and the minimum is {generated_image.min():.4f}")
print(
    f"The mean value of the pixels is {generated_image.mean():.4f} and the standard deviation is {generated_image.std():.4f}\n")

sns.histplot(generated_image.ravel(),
             label=f"Pixel Mean {np.mean(generated_image):.4f} & Standard Deviation {np.std(generated_image):.4f}",
             kde=False)
plt.legend(loc='upper center')
plt.title('Distribution of Pixel Intensities in the Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('# Pixels in Image')
plt.show()

#---------------- CNN model -----------------


# calculate class weight
weight_for_0 = num_pneumonia / (num_normal + num_pneumonia)  # weight for class 0 (Pneumonie)
weight_for_1 = num_normal / (num_normal + num_pneumonia)  # weight for class 1 (normal)

class_weight = {0: weight_for_0, 1: weight_for_1}  # class weights saved for a training in a directory

print(f"Weight for class 0: {weight_for_0:.2f}")
print(f"Weight for class 1: {weight_for_1:.2f}")

#---------------- definition and compilation of the CNN-model -----------------

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten, BatchNormalization, Input

# definition of the model: a sequential model with multiple layers
model = Sequential()

model.add(Input(shape=(180, 180, 3)))  # input layer defined
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))  # first convolution layer
model.add(BatchNormalization())  # Batch-Normalisierung
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))  # second convolution layer
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))  # Max-Pooling-Schicht

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

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print("\n\n +++ Model Training")
model.summary()

#---------------- model training -----------------

# perform training
r = model.fit(
    train,  # trainings data
    epochs=10,  # amount of epochs (10 full passes through trainings data)
    validation_data=validation,  # validation data
    class_weight=class_weight,  # class weights to compensate for class imbalance
    steps_per_epoch=100,  # amount of steps per epoch
    validation_steps=25,  # amount of steps for validation steps per epoch
)

# epoch: a full pass through the whole trainings set
# training model with over 10 epochs with the given training and validation data
# class weight used to compensate for imbalanced weights
# steps_per_epoch and validation_steps set how many batch steps per epoch and validations are executed


#---------------- visualize training results -----------------

plt.figure(figsize=(12, 8))

# plot loss function per epoch
plt.subplot(2, 2, 1)
plt.plot(r.history['loss'], label='Loss')
plt.plot(r.history['val_loss'], label='Val_Loss')
plt.legend()
plt.title('Loss Evolution')
plt.show()

# plot accuracy per epoch
plt.subplot(2, 2, 2)
plt.plot(r.history['accuracy'], label='Accuracy')
plt.plot(r.history['val_accuracy'], label='Val_Accuracy')
plt.legend()
plt.title('Accuracy Evolution')
plt.show()

#---------------- model evaluation -----------------

# evaluate test data
evaluation = model.evaluate(test)
print(f"Test Accuracy: {evaluation[1] * 100:.2f}%")

# evaluate training data
evaluation = model.evaluate(train)
print(f"Train Accuracy: {evaluation[1] * 100:.2f}%")

#---------------- model evaluation with additional metrics -----------------


from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

# prediction of the test data
pred = model.predict(test)

# calculate and output confusion matrix
print(confusion_matrix(test.classes, pred > 0.5))

#---------------- with confusion matrix and classifications report -----------------

# classification report created and displayed as DataFrame
print(confusion_matrix(test.classes, pred > 0.5))
pd.DataFrame(classification_report(test.classes, pred > 0.5, output_dict=True))

# confusion matrix with threshold 0.7 calculated and displayed
print(confusion_matrix(test.classes, pred > 0.7))
# classification report with threshold of 0.7 calculated and displayed as dataframe
pd.DataFrame(classification_report(test.classes, pred > 0.7, output_dict=True))

#----------transfer Learning------------
#----DenseNET----

from keras.applications.densenet import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras import backend as K

# load pre-trained DenseNet121-model, omitting the highest layer and use weights from ImageNet
base_model = DenseNet121(input_shape=(180, 180, 3), include_top=False, weights='imagenet', pooling='avg')

# display architecture of basemodel
base_model.summary()

# output amount of layers in basemodel
layers = base_model.layers
print(f"\nThe model has {len(layers)} layers")

# output input and output form of the basemodel
print(f"\nThe input shape {base_model.input}")
print(f"The output shape {base_model.output}\n")

# create new model by adding layers to the basemodel
#model = Sequential()
base_model = DenseNet121(include_top=False, weights='imagenet')
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Global Average Pooling-Schicht hinzufügen
predictions = Dense(1, activation="sigmoid")(
    x)  # Dense-Schicht mit Sigmoid-Aktivierung für binäre Klassifikation hinzufügen

# the full model defined
model = Model(inputs=base_model.input, outputs=predictions)
#model.add(base_model)
#model.add(GlobalAveragePooling2D())
#model.add(Dense(1, activation='sigmoid'))

# model of binary crossentropy loss, adam optimizer and compile accuracy matrix
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# model trained with training data, class weights and validation data used

r = model.fit(
    train,
    epochs=10,
    validation_data=validation,
    class_weight=class_weight,
    steps_per_epoch=100,
    validation_steps=25,
)

# plot training result
plt.figure(figsize=(12, 8))

# plot loss development
plt.subplot(2, 2, 1)
plt.plot(r.history['loss'], label='Loss')
plt.plot(r.history['val_loss'], label='Val_Loss')
plt.legend()
plt.title('Loss Evolution')
plt.show()

# plot accuracy development
plt.subplot(2, 2, 2)
plt.plot(r.history['accuracy'], label='Accuracy')
plt.plot(r.history['val_accuracy'], label='Val_Accuracy')
plt.legend()
plt.title('Accuracy Evolution')
plt.show()

# evaluate model with test data
evaluation = model.evaluate(test)
print(f"\nTest Accuracy: {evaluation[1] * 100:.2f}%")

# evaluate model with train data
evaluation = model.evaluate(train)
print(f"Train Accuracy: {evaluation[1] * 100:.2f}%")

#--------evaluation----------
predicted_vals = model.predict(test, steps=len(test))

print(confusion_matrix(test.classes, predicted_vals > 0.5))
pd.DataFrame(classification_report(test.classes, predicted_vals > 0.5, output_dict=True))
