import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Activation,Conv1D,Dropout,MaxPooling1D,Flatten
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

train = pd.read_csv("C:/Users/furka/CNNProject/train.csv")
test = pd.read_csv("C:/Users/furka/CNNProject/test.csv")

label_encoder = LabelEncoder().fit(train.species)
labels = label_encoder.transform(train.species)
classes = list(label_encoder.classes_)

train = train.drop(["id","species"],axis=1)
test = test.drop(["id"],axis=1)
nb_features = 192
nb_classes = len(classes)

scaler = StandardScaler().fit(train.values)
train = scaler.transform(train.values)

X_train, X_valid, y_train, y_valid = train_test_split(train,labels, test_size=0.1)

y_train = to_categorical(y_train)
y_valid = to_categorical(y_valid)

X_train = np.array(X_train).reshape(891,192,1)
X_valid = np.array(X_valid).reshape(99,192,1)

model = Sequential()
model.add(Conv1D(512,1,input_shape=(nb_features,1)))
model.add(Activation("relu"))
model.add(MaxPooling1D(2))
model.add(Conv1D(256,1))
model.add(MaxPooling1D(2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(2048, activation="relu"))
model.add(Dense(1024, activation="relu"))
model.add(Dense(nb_classes, activation="softmax"))
model.summary()

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(X_train, y_train, epochs=30, validation_data=(X_valid,y_valid))

print(("Ortalama Eğitim Kaybı: ", np.mean(model.history.history["loss"])))
print(("Ortalama Eğitim Başarımı: ", np.mean(model.history.history["accuracy"])))
print(("Ortalama Doğrulama Kaybı: ", np.mean(model.history.history["val_loss"])))
print(("Ortalama Doğrulama Başarımı: ", np.mean(model.history.history["val_accuracy"])))

plt.plot(model.history.history['loss'], color='g', label="Eğitim kaybı")
plt.plot(model.history.history['val_loss'], color='y', label="Doğrulama kaybı")
plt.legend()
plt.show()

plt.plot(model.history.history['accuracy'], color='b', label="Eğitim başarımı")
plt.plot(model.history.history['val_accuracy'], color='r', label="Doğrulama başarımı")
plt.legend()
plt.show()