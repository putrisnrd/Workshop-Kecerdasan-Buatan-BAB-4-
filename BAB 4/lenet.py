from keras.layers import (
    Activation,
    AveragePooling2D,
    Conv2D,
    Dense,
    Flatten,
    Input,
)
from keras.models import Sequential

model = Sequential()

# 1. Tambahkan Input layer secara eksplisit (Menghilangkan Warning)
model.add(Input(shape=(28, 28, 1)))  # Sesuaikan dengan ukuran citra input Anda

# 2. Layer Conv 1 & AveragePooling 1 (Perbaikan TypeError)
model.add(Conv2D(6, kernel_size=(5, 5), activation="tanh", padding="same"))
model.add(AveragePooling2D(pool_size=(2, 2)))  # Wajib isi pool_size=(2, 2)

# 3. Layer Conv 2 & AveragePooling 2
model.add(
    Conv2D(16, kernel_size=(5, 5), activation="tanh", padding="valid")
)
model.add(AveragePooling2D(pool_size=(2, 2)))  # Wajib isi pool_size=(2, 2)

# 4. Dense Layers (FC)
model.add(Flatten())
model.add(Dense(120, activation="tanh"))
model.add(Dense(84, activation="tanh"))
model.add(Dense(10, activation="softmax"))  # 10 kelas output

model.summary()