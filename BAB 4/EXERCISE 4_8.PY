import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from keras.models import Sequential
from keras.utils import to_categorical

# Load & Preprocess Data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype("float32") / 255.0
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype("float32") / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]

# Build Model Exercise 4.8
model = Sequential()
model.add(Input(shape=(28, 28, 1)))

# Blok Conv 1
model.add(Conv2D(32, (5, 5), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# MODIFIKASI EXERCISE 4.8: Blok Conv 2 Tambahan
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# Fully Connected Layers
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(num_classes, activation="softmax"))

model.compile(
    loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)
model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=200,
)

scores = model.evaluate(X_test, y_test, verbose=0)
print("EXERCISE 4.8 - CNN Error: %.2f%%" % (100 - scores[1] * 100))