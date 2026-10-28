# Example 4.5 AlexNet dengan Keras (Diperbarui untuk Keras 3 / TF 2.x)

import keras
from keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
)
from keras.models import Sequential

# Inisialisasi Model AlexNet
model = Sequential()

# Layer Input Eksplisit
model.add(Input(shape=(224, 224, 3)))

# 1st Convolutional Layer
model.add(
    Conv2D(
        filters=96,
        kernel_size=(11, 11),
        strides=(4, 4),
        padding="valid",
        activation="relu",
    )
)
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid"))

# 2nd Convolutional Layer
model.add(
    Conv2D(
        filters=256,
        kernel_size=(5, 5),
        strides=(1, 1),
        padding="same",
        activation="relu",
    )
)
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid"))

# 3rd Convolutional Layer
model.add(
    Conv2D(
        filters=384,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        activation="relu",
    )
)
model.add(BatchNormalization())

# 4th Convolutional Layer
model.add(
    Conv2D(
        filters=384,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        activation="relu",
    )
)
model.add(BatchNormalization())

# 5th Convolutional Layer
model.add(
    Conv2D(
        filters=256,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        activation="relu",
    )
)
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid"))

# Flatten & Fully Connected Layers
model.add(Flatten())

# 1st Fully Connected Layer
model.add(Dense(4096, activation="relu"))
model.add(Dropout(0.4))

# 2nd Fully Connected Layer
model.add(Dense(4096, activation="relu"))
model.add(Dropout(0.4))

# Output Layer (17 Kelas, misalnya Oxford 17 Flowers Dataset)
model.add(Dense(17, activation="softmax"))

# Tampilkan Ringkasan Arsitektur Model
model.summary()

# Kompilasi Model
model.compile(
    loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)