from keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from keras.models import Sequential

model = Sequential()

# 1st Convolutional Layer
model.add(Input(shape=(224, 224, 3)))
model.add(
    Conv2D(
        filters=96,
        kernel_size=(11, 11),
        strides=(4, 4),
        activation="relu",
        padding="valid",
    )
)
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))

# 2nd Convolutional Layer
model.add(
    Conv2D(
        filters=256,
        kernel_size=(5, 5),
        strides=(1, 1),
        activation="relu",
        padding="same",
    )
)
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))

# 3rd Convolutional Layer
model.add(
    Conv2D(
        filters=384,
        kernel_size=(3, 3),
        strides=(1, 1),
        activation="relu",
        padding="same",
    )
)

# 4th Convolutional Layer
model.add(
    Conv2D(
        filters=384,
        kernel_size=(3, 3),
        strides=(1, 1),
        activation="relu",
        padding="same",
    )
)

# 5th Convolutional Layer
model.add(
    Conv2D(
        filters=256,
        kernel_size=(3, 3),
        strides=(1, 1),
        activation="relu",
        padding="same",
    )
)
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))

# Fully Connected Layers
model.add(Flatten())

# MODIFIKASI EXERCISE 4.5: Mengubah unit Dense Layer (diubah dari 4096, 4096, 1000)
model.add(Dense(1024, activation="relu"))  # 1st Fully Connected
model.add(Dropout(0.4))

model.add(Dense(1024, activation="relu"))  # 2nd Fully Connected
model.add(Dropout(0.4))

model.add(Dense(512, activation="relu"))  # 3rd Fully Connected
model.add(Dropout(0.4))

# Output Layer
model.add(Dense(17, activation="softmax"))

# Tampilkan ringkasan model
model.summary()