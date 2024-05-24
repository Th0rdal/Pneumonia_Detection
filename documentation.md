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


