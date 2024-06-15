import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import global_var


#---------------- Image Pre Processing -----------------
def imagePreProcessing():
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
    global_var.train = image_generator.flow_from_directory(global_var.train_dir,
                                                           batch_size=8,
                                                           # Number of images to process at a time / # Images per batch
                                                           shuffle=True,
                                                           class_mode='binary',
                                                           target_size=(
                                                           180, 180))  # Resize the images to 180x180 pixels

    global_var.validation = image_generator.flow_from_directory(global_var.val_dir,
                                                                batch_size=1,
                                                                shuffle=False,
                                                                class_mode='binary',
                                                                target_size=(180, 180))

    global_var.test = image_generator.flow_from_directory(global_var.test_dir,
                                                          batch_size=1,
                                                          shuffle=False,
                                                          class_mode='binary',
                                                          target_size=(180, 180))

    # check the pixel value of the presentation
    generated_image, label = global_var.train.__getitem__(0)
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
