import os

os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow
import tensorflow as tf
from IPython.display import Image, display
import global_var


#---------------- Data Visualization: DATASET in numbers -----------------
def dataVisualization():
    # print amount of data in console
    print("========================================\nTrain set:")
    print(f"PNEUMONIA={global_var.num_pneumonia}")
    print(f"NORMAL={global_var.num_normal}")
    print(" ")

    print("========================================\nTest set:")
    print(
        f"PNEUMONIA={len(os.listdir(os.path.join(global_var.test_dir, 'PNEUMONIA')))}")  # counts amount of files in given path
    print(
        f"NORMAL={len(os.listdir(os.path.join(global_var.test_dir, 'NORMAL')))}")  # counts amount of files in given path
    print(" ")

    print("========================================\nValidation set:")
    print(
        f"PNEUMONIA={len(os.listdir(os.path.join(global_var.val_dir, 'PNEUMONIA')))}")  #counts amount of files in given path
    print(
        f"NORMAL={len(os.listdir(os.path.join(global_var.val_dir, 'NORMAL')))}")  #counts amount of files in given path
    print(" ")

    #---------------- Data Visualization: 9x9 PNEUMONIA -----------------

    pneumonia_files = [file for file in os.listdir(global_var.pneumonia_dir) if
                       file.lower().endswith(('.png', '.jpg', '.jpeg'))]
    # creates list of filenames from pneumonia_dir with .jpg, -png or .jpeg ending

    plt.figure(figsize=(20, 10))  # create 20x10 plot
    for i in range(min(9, len(pneumonia_files))):  # max 9 pictures are shown in plot
        img_path = os.path.join(global_var.pneumonia_dir, pneumonia_files[i])
        img = plt.imread(img_path)
        plt.subplot(3, 3, i + 1)
        plt.imshow(img, cmap='gray')  # pictures in greyscale
        plt.axis('off')  # no axis label

    plt.suptitle('Train/Pneumonia', fontsize=32)  # sets title of figure
    plt.tight_layout(rect=(0, 0, 1, 0.95))  # fits layout to make space for subtitle
    plt.show()

    #---------------- Data Visualization: 9x9 NORMAL -----------------

    normal_files = [file for file in os.listdir(global_var.normal_dir) if
                    file.lower().endswith(('.png', '.jpg', '.jpeg'))]
    # creates list of filenames from normal_dir with .jpg, .png or .jpeg ending

    plt.figure(figsize=(20, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        img = plt.imread(os.path.join(global_var.normal_dir, normal_files[i]))
        plt.imshow(img, cmap='gray')
        plt.axis('off')

    plt.suptitle('Train/Normal', fontsize=32)  # sets title of figure
    plt.tight_layout(rect=(0, 0, 1, 0.95))  # fits layout to make space for subtitle
    plt.show()

    #---------------- Data Visualization: SAMPLE IMAGE ----------------

    # sampe image: shows picture as sample
    pic_nr = 15  # file that should be used as sample image
    pic_path = "resources/input/chest_xray/train/NORMAL"
    normal_img = os.listdir(pic_path)[pic_nr]
    sample_img = plt.imread(os.path.join(global_var.normal_dir, normal_img))
    plt.imshow(sample_img, cmap='gray')  # show image in greyscale
    plt.colorbar()
    plt.title('Raw Chest X Ray Image')  # title of the Plots

    print(f"Sample Picture {pic_nr} loaded from {pic_path}")
    print(
        f"The dimensions of the image are {sample_img.shape[0]} pixels width and {sample_img.shape[1]} pixels height, one single color channel.")
    print(f"The maximum pixel value is {sample_img.max():.4f} and the minimum is {sample_img.min():.4f}")
    print(
        f"The mean value of the pixels is {sample_img.mean():.4f} and the standard deviation is {sample_img.std():.4f}")

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


#---------------- visualize training results -----------------
def baseTrainingResultVisualization(r, model):
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
    evaluation = model.evaluate(global_var.test)
    print(f"Test Accuracy: {evaluation[1] * 100:.2f}%")

    # evaluate training data
    evaluation = model.evaluate(global_var.train)
    print(f"Train Accuracy: {evaluation[1] * 100:.2f}%")


def visualizeGradCAM(model, name="", path=None, pred_index=None):
    if path is None or path == "":
        path = "resources/input/chest_xray/test/NORMAL/IM-0001-0001.jpeg"

    model.summary()
    layer = input("Enter the layer you want to create the gradCAM for: ")

    preprocess_input = keras.applications.xception.preprocess_input

    model.layers[-1].activation = None

    # Load and preprocess the image
    img_path = path
    img = keras.utils.load_img(img_path, target_size=(180, 180))
    array = keras.utils.img_to_array(img)

    array = np.expand_dims(array, axis=0)
    array = preprocess_input(array)

    preds = model.predict(array)
    preds = np.array(preds)  # Convert to NumPy array if not already
    preds = preds.reshape((1, -1))
    print("Raw prediction:", preds)

    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(layer).output, model.output]
    )

    # Compute the gradient of the class output with respect to the feature map
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(array)
        if pred_index is None:
            pred_index = tensorflow.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Gradient of the output neuron with respect to the output feature map
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Compute the mean intensity of the gradients over all the filters
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel in the feature map array by the importance of the channel
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU to the heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    # Display the heatmap
    plt.matshow(heatmap)
    plt.title("Heatmap " + name)
    plt.show()

    # Superimpose the heatmap on the original image
    heatmap = np.uint8(255 * heatmap)
    jet = mpl.colormaps["jet"]
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((180, 180))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * 0.4 + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)

    plt.imshow(superimposed_img)
    plt.title("Superimposed Image " + name)
    plt.axis('off')
    plt.show()
