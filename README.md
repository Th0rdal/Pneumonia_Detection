# 1. Dataset auslesen & preprocessen
## 1.1 Konstruktor
```python
class ReadDataset:
    def __init__(self, datasetpath, labels, image_shape):
        self.datasetpath = datasetpath
        self.labels = labels
        self.image_shape = image_shape
```

* __init__ ist der Konstruktor einer Klasse in Python

```python
readDatasetObject = ReadDataset('/input/chest-xray-pneumonia/chest_xray/train',
                                ['NORMAL', 'PNEUMONIA'],
                                (64, 64))

```
* Konstruktor wird hier aufgerufen
* Es wird angegeben:
  * datasetpath für die Trainingsdaten
  * Die zwei Labels NORMAL & PNEUMONIA
  * der image_shape, also die Zielgröße der Bilder nach dem preprocessing

## 1.2 readImages()

```python
def readImages(self):
    self.returListImages()  
    self.finalImages = []  
    labels = []  
    for label in range(len(self.labels)):  # Iteriert über die Label-Klassen
        for img in self.images[label]:  # Iteriert über die Bilder in jeder Label-Klasse
            img = cv2.imread(str(img))  # Liest das Bild ein
            img = cv2.resize(img, self.image_shape)  # Skaliert das Bild auf die angegebene Größe
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Konvertiert das Bild von BGR zu RGB
            img = img / 255.0  # Normalisiert die Pixelwerte (0 bis 1)
            self.finalImages.append(img)  # Fügt das verarbeitete Bild zur Liste hinzu
            labels.append(label)  # Fügt das entsprechende Label zur Label-Liste hinzu
    images = np.array(self.finalImages)  
    labels = np.array(labels)  
    return images, labels  # Gibt die Bilder und Labels zurück

```
* Jedes Bild wird mit cv2.imread eingelesen
* Das Bild wird mit cv2.resize auf die Größe self.image_shape, also hier 64x64 skaliert
* Das Bild wird mit cv2.cvtColor von BGR (Standardformat von OpenCV) zu RGB konvertiert
  * wird benötigt um Probleme mit TensorFlow, Keras, Matplotlib zu vermeiden, diese verwenden RGB, laut GPT
* Die Pixelwerte des Bildes werden durch 255 geteilt, um sie auf den Bereich 0 bis 1 zu normalisieren
