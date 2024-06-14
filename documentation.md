# Dokumentation des Codes #
## Data Visualisation ##
### Code Block 1 ###
Dieser Code Block gibt die Anzahl der Files in den Training-Ordnern in der Konsole aus.
```python
#Anzahl der Daten in Konsole ausgeben
print("========================================\nTrain set:")
num_pneumonia = len(os.listdir(os.path.join(train_dir, 'PNEUMONIA'))) #zählt die anzahl der dateien im gegebenen pfad
num_normal = len(os.listdir(os.path.join(train_dir, 'NORMAL')))
print(f"PNEUMONIA={num_pneumonia}")
print(f"NORMAL={num_normal}")
print(" ")

```

1) `num_pneumonia = len(os.listdir(os.path.join(train_dir, 'PNEUMONIA')))`

`os.path.join(path1,path2)` => fügt zwei Pfade zu einem zusammen und gibt ihn zurück. Hier wird einfach zum Pfad zum trainingsdirectory noch das Verzeichnis PNEUMONIA angehängt.

`os.listdir(path)` => gibt eine Liste von allen Files und Directories in gegebenen Path zurück

`len(list)` => gibt die Länge einer Liste zurück

------------

### Code Block 2 ###
Dieser Code Block zeigt die ersten 9 Bilder vom Pfad `input/chest_xray/train/PNEUMONIA` an in einem Plot an.
```python
#---------------- Data Visualization: PNEUMONIA -----------------

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
```
1) `pneumonia_files = [file for file in os.listdir(pneumonia_dir) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]`

`os.listdir(pneumonia_dir)`=> gibt eine Liste aller Dateien und Verzeichnisse im angegeben Pfad zurück.

`file for file in os.listdir()`=> iteriert durch jedes Element der Liste die durch `os.listdir()` zurückgegeben wird.

`if file.lower().endswith(('.png', '.jpg', '.jpeg'))` wenn das aktuelle element in dem gerade iteriert ist eines der gegeben file-enden hat ist die Bedingung erfüllt.

Wenn die bedingung für ein Element in der Liste erfüllt ist, dann wird es in die Liste `pneumonia_files` hinzugefügt

2) `plt.figure(figsize=(20, 10))` => Erstellt einen Plot mit den Maßen 20x10


3) `for i in range(min(9, len(pneumonia_files))):`

`len(pneumonia_files)` => gibt Anzahl der Elemente in der Liste zurück

`min(9, len(pneumonia_files))` => gibt den kleineren Wert von 9 und len(pneumonia_files) zurück.

Wenn `len(pneumonia_files)` kleiner oder gleich 9 ist, gibt `min()` diesen Wert zurück.
Wenn `len(pneumonia_files)` größer als 9 ist, gibt `min()` den Wert 9 zurück.

Das bedeutet dass die Schleife 9 oder weniger Bilder anzeigt. Wenn es keine 9 Bilder im Directory gibt werden so viele angezeigt wie es gibt. Wenn es 9 oder mehr gibt, werden 9 angezeigt.

4) Image auslesen
```python
img_path = os.path.join(pneumonia_dir, pneumonia_files[i])
img = plt.imread(img_path)
```
Hier wird zuerst das der eindeutige Pfad für das Image in der aktuellen Iteration kombiniert. In der zweiten Zeile wird dann dieses Image in die Variable `img` eingelesen.

5) Image in Subplot speichern
```python
plt.subplot(3, 3, i + 1)
plt.imshow(img, cmap='gray')
plt.axis('off')
```

`plt.subplot(3, 3, i + 1):` =>
Erstellt ein Subplot-Raster mit 3x3 und wählt das i+1-te Subplot für das aktuelle Bild aus. Legt fest, wo das Bild in der Figur angezeigt wird.

`plt.imshow(img, cmap='gray')` => Zeigt das Bild img im aktuellen Subplot an. Das Argument `cmap='gray'` legt fest dass es in Graustufen angezeigt wird.

`plt.axis('off'):` => Schaltet die Achsenbeschriftungen und -markierungen des aktuellen Subplots au

6) Anzeige des Plots

```python
plt.tight_layout()
plt.show()
```
`plt.tight_layout():` => Passt die Subplots so an, dass sie sich nicht überlappen und der verfügbare Platz komplett genutzt wird.

`plt.show():` => Zeigt den Plot mit den Subplots an.

-------------------

### Code Block 3 ###
Mit diesem Code wird ein Sample Picture angezeigt von dem im nächsten Code Block 4 dann auch ein Grapgh der Verteilung der Farbwerte (0 - 255) gezeigt wird.

```python
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
```

1) Sample Image auslesen
```python
pic_nr = 15
pic_path = "input/chest_xray/train/NORMAL"
normal_img = os.listdir(pic_path)[pic_nr]
sample_img = plt.imread(os.path.join(normal_dir, normal_img))
```

`pic_nr = 15`, `normal_img = os.listdir(pic_path)[pic_nr]` => das 15. Element (Name der Datei) aus der Liste die durch `os.listdir()` generiert wird, wird in `normal_img`gespeichert.

`sample_img = plt.imread(os.path.join(normal_dir, normal_img))` => dieses 15. File in dem Directory wird ausgelesen. 

2) Image in Plot anzeigen

```python
plt.imshow(sample_img, cmap='gray')
plt.colorbar()
plt.title('Raw Chest X Ray Image') #Titel des Plots
```

`plt.imshow(sample_img, cmap='gray')` => Zeigt das Bild sample_img im aktuellen Subplot an. Das Argument `cmap='gray'` legt fest dass es in Graustufen angezeigt wird.

`plt.colorbar():` => fügt dem Plot eine Farbskala hinzu. In diesem Fall Graustufen.

`plt.title('Raw Chest X Ray Image')` => Titel des Plots der angezeigt wird.

-----

### Code Block 4 ###
Mit diesem Code wird ein Plot erstellt der die "Distribution of Pixel Intensities in the Image" anzeigt.

```python
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
```

1) `plt.figure(figsize=(10, 6))` => Erstellt einen Plot mit 10x6


2) `sns.histplot(sample_img.ravel(), kde=False, bins=32, label=f"Pixel Mean {np.mean(sample_img):.4f} & Standard Deviation {np.std(sample_img):.4f}")`


`sns.histplot(...):` => Funktion von Seaborn erstellt ein Histogramm.

`sample_img.ravel():`=> `ravel()`wandelt das 2D-Array `sample_img` in ein 1D-Array um. Ist notwendig, da `histplot` die Verteilung einer eindimensionalen Datenmenge darstellt.

`kde=False:` => Nur Hostogramm wird angezeigt, keine KDE Kurve. Was auch immer das ist haha.

`bins=32:` =>  legt die Anzahl der Bins (Intervalle) im Histogramm auf 32 fest. Bins sind die Intervalle, in die die Daten unterteilt werden auf der x-Achse.

`label=f"Pixel Mean {np.mean(sample_img):.4f} & Standard Deviation {np.std(sample_img):.4f}":` => `label` fügt dem Plot eine Beschriftung hinzu. Hier wird ein f-stringS verwendet, um den Mittelwert und die Standardabweichung der Pixelintensitäten anzuzeigen.

3. 
```python
plt.legend(loc='upper center') #legende (label mit Pixel Mean & Std. Deviation) wird 
# in upper center des Bildes angezeigt.
plt.title('Distribution of Pixel Intensities in the Image') #title
plt.xlabel('Pixel Intensity') #label der x-Achse
plt.ylabel('# Pixels in Image') #label der y-Achse
plt.xlim(0, 255)  #begrenzung der x-Werte auf 0 - 255 (Graustufen)
plt.show() #plot anzeigen
```










# stella: kurzer dump an Informationen 


Medical Diagnosis with CNN and Transfer Learning

The code below implements two distinct models for medical image classification using Convolutional Neural Networks (CNN) and transfer learning with DenseNet121. The goal is to train and evaluate these models on chest X-ray images to classify them into categories such as PNEUMONIA and NORMAL. The task involves configuring various hyperparameters and experimenting with different network architectures.

Model 1: Custom CNN Model

This model is built from scratch using a sequential approach with several convolutional layers, batch normalization, max-pooling layers, and dense layers with dropout for regularization.

Model 2: Transfer Learning with DenseNet121

This model leverages transfer learning by utilizing a pre-trained DenseNet121 model. We add a global average pooling layer and a dense layer with sigmoid activation to customize it for our binary classification task.