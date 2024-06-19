import tensorflow as tf
from keras.datasets import mnist
import cv2  #module: opencv-python 4.10.0.82 (neueste pip version muss installiert sein)
import os
import pathlib
from keras.layers import Conv2D, Conv2DTranspose, Dropout, Dense, Reshape, LayerNormalization, LeakyReLU
from keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

import pydot
from tensorflow.keras.utils import plot_model

from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import f1_score, recall_score, precision_score


# ----------------------------------------------------------------------

class ReadDataset:
    def __init__(self, datasetpath, labels, image_shape):
        self.datasetpath = datasetpath
        self.labels = labels
        self.image_shape = image_shape

    def returListImages(self, ):
        self.images = []
        for label in self.labels:
            self.images.append(list(pathlib.Path(os.path.join(self.datasetpath,
                                                              label)).glob('*.*')))

    def readImages(self, ):
        self.returListImages()
        self.finalImages = []
        labels = []
        for label in range(len(self.labels)):
            for img in self.images[label]:
                img = cv2.imread(str(img))
                img = cv2.resize(img, self.image_shape)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img / 255.0
                self.finalImages.append(img)
                labels.append(label)
        images = np.array(self.finalImages)
        labels = np.array(labels)
        return images, labels


readDatasetObject = ReadDataset('input/chest_xray/train',
                                ['NORMAL', 'PNEUMONIA'],
                                (180, 180))  #180 x 180 eigentlich bei unserem Recognition Model
images, labels = readDatasetObject.readImages()
print(f'Images: Amount={images.shape[0]}, {images.shape[1]}x{images.shape[2]}, Channels={images.shape[3]}')
print(f'Labels: Amount={labels.shape[0]}')

# ----------------------------------------------------------------------

plt.figure(figsize=(12, 12))
indexs = np.random.randint(0, len(labels), size=(64,))
for i in range(64):
    plt.subplot(8, 8, (i + 1))
    plt.imshow(images[indexs[i]])
    plt.title(labels[indexs[i]])
plt.tight_layout()  # Optional: automatic tight Layout for better space usage
plt.show()  # show the generated plot


# ----------------------------------------------------------------------
class Acgan:
    def __init__(self, eta, batch_size, epochs, weight_decay, latent_space,
                 image_shape, kernel_size):
        self.eta = eta
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.latent_space = latent_space
        self.image_shape = image_shape
        self.kernel_size = kernel_size

    def data(self, images, labels):
        ytrain = tf.keras.utils.to_categorical(labels)
        self.images = images
        self.labels = ytrain

    def samples(self, G, noize, labels):
        images = G.predict([noize, labels])
        ys = np.argmax(labels, axis=1)
        plt.figure(figsize=(12, 4))
        for i in range(16):
            plt.subplot(2, 8, (i + 1))
            plt.imshow(images[i], cmap='gray')
            plt.title(ys[i])
        plt.show()

    def generator(self, inputs, labels):
        filters = [256, 128, 64, 32]
        padding = 'same'
        x = inputs
        y = labels
        x = layers.concatenate([x, y])
        x = layers.Dense(1024, )(x)
        x = layers.Dense(8 * 8 * filters[0],
                         kernel_regularizer=tf.keras.regularizers.L2(0.001))(x)
        x = layers.Reshape((8, 8, filters[0]))(x)
        for filter in filters:
            if filter >= 64:
                strides = 2
            else:
                strides = 1
            x = LayerNormalization()(x)
            x = layers.Activation('relu')(x)
            x = Conv2DTranspose(filter, kernel_size=self.kernel_size, padding=padding,
                                strides=strides)(x)
        x = Conv2DTranspose(3, kernel_size=self.kernel_size, padding=padding)(x)
        x = layers.Activation('sigmoid')(x)
        self.generatorModel = models.Model(inputs=[inputs, labels],
                                           outputs=x,
                                           name='generator')

    def discriminator(self, inputs):
        x = inputs
        filters = [32, 64, 128, 256]
        padding = 'same'
        for filter in filters:
            if filter < 256:
                strides = 2
            else:
                strides = 1
            x = Conv2D(filter, kernel_size=self.kernel_size, padding=padding,
                       strides=strides,
                       kernel_regularizer=tf.keras.regularizers.L2(0.001))(x)
            x = LeakyReLU(alpha=0.2)(x)
        x = layers.Flatten()(x)
        outputs = Dense(1, )(x)
        labelsOutput = Dense(256,
                             kernel_regularizer=tf.keras.regularizers.L2(0.001))(x)
        labelsOutput = Dropout(0.3)(labelsOutput)
        labelsOutput = Dense(2, )(labelsOutput)
        labelsOutput = layers.Activation('softmax')(labelsOutput)
        self.discriminatorModel = models.Model(inputs=inputs,
                                               outputs=[outputs, labelsOutput],
                                               name='discriminator')

    def build(self):
        generatorInput = layers.Input(shape=(self.latent_space,))
        discriminatorInput = layers.Input(shape=self.image_shape)
        labelsInput = layers.Input(shape=(2,))

        self.generator(generatorInput, labelsInput)
        self.discriminator(discriminatorInput)

        G = self.generatorModel
        D = self.discriminatorModel

        # Diskriminator muss trainierbar sein
        D.trainable = True
        D.compile(loss=['mse', 'binary_crossentropy'],
                  optimizer=tf.keras.optimizers.RMSprop(learning_rate=self.eta,
                                                        weight_decay=self.weight_decay))
        D.summary()
        G.summary()

        D.trainable = False
        GAN = models.Model(inputs=[generatorInput, labelsInput],
                           outputs=D(G([generatorInput, labelsInput])))
        GAN.compile(loss=['mse', 'binary_crossentropy'],
                    optimizer=tf.keras.optimizers.RMSprop(learning_rate=self.eta * 0.5,
                                                          weight_decay=self.weight_decay * 0.5))
        GAN.summary()
        return G, D, GAN

    def trainAlgorithm(self, G, D, GAN):
        for epoch in range(self.epochs):
            indexs = np.random.randint(0, len(self.images), size=self.batch_size)
            realImages = self.images[indexs]
            realLabels = self.labels[indexs]
            realTag = tf.ones(shape=(self.batch_size,))

            noize = tf.random.uniform(shape=(self.batch_size, self.latent_space), minval=-1, maxval=1)
            fakeLabels = tf.keras.utils.to_categorical(np.random.choice(range(2), size=self.batch_size), num_classes=2)
            fakeImages = tf.squeeze(G.predict([noize, fakeLabels], verbose=0))
            fakeTag = tf.zeros(shape=(self.batch_size,))

            allImages = np.vstack([realImages, fakeImages])
            allLabels = np.vstack([realLabels, fakeLabels])
            allTags = np.hstack([realTag, fakeTag])

            # Diskriminator muss trainierbar sein
            D.trainable = True
            d_loss = D.train_on_batch(allImages, [allTags, allLabels])

            noize = tf.random.uniform(shape=(self.batch_size, self.latent_space), minval=-1, maxval=1)

            # GAN trainieren (Diskriminator nicht trainierbar)
            D.trainable = False
            g_loss = GAN.train_on_batch([noize, fakeLabels], [realTag, fakeLabels])

            if epoch % 10 == 0:  #5000
                print(f'Epoch: {epoch}')
                print(f'discriminator loss: {d_loss}, generator loss: {g_loss}')
                self.samples(G, noize, fakeLabels)


acgan = Acgan(eta=0.0001, batch_size=64, epochs=120, weight_decay=6e-9, latent_space=100, image_shape=(180, 180, 3),
              kernel_size=5)
# standardmäßig: 32000 Epochs

acgan.data(images, labels)

G, D, GAN = acgan.build()

# tf.keras.utils.plot_model(GAN, show_shapes = True)
# tf.keras.utils.plot_model(D, show_shapes = True)
# tf.keras.utils.plot_model(G, show_shapes = True)

acgan.trainAlgorithm(G, D, GAN)
G.save('/kaggle/working/generator.h5')

#------------------------------ GENERIEREN von Bildern, mit dem Model ------------------------------

datasetGenerationSize = 500
noize = tf.random.uniform(shape=(datasetGenerationSize, 100), minval=-1, maxval=1)
newlabels = tf.keras.utils.to_categorical(np.random.choice([0, 1], size=(datasetGenerationSize,)), num_classes=2)

print(noize.shape)
print(newlabels.shape)

np.unique(np.argmax(newlabels, axis=1), return_counts=True)

imagesGeneration = G.predict([noize, newlabels])
print(imagesGeneration.shape)

plt.figure(figsize=(12, 12))
t = np.argmax(newlabels, axis=1)
for i in range(64):
    plt.subplot(8, 8, (i + 1))
    plt.imshow(imagesGeneration[i])
    plt.title(t[i])
plt.show()

basemodel = tf.keras.applications.VGG16(weights=None, input_shape=(180, 180, 3),
                                        pooling='max', include_top=False)
x = layers.Dropout(0.4)(basemodel.output)
x = layers.Dense(128, )(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.2)(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(32, )(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.2)(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(1, activation='sigmoid')(x)
m = tf.keras.models.Model(inputs=basemodel.input, outputs=x)
m.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001))
m.summary()

history = m.fit(imagesGeneration, np.argmax(newlabels, axis=1),
                epochs=60, batch_size=64,
                validation_split=0.2,
                callbacks=[tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss', mode='min',
                                                            restore_best_weights=True)])

plt.figure(figsize=(7, 6))
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.title('Results obtained while training a neural network on images generated by the neural network')
plt.show()
