import time
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, Input
from keras.models import Sequential

# 1. Menyiapkan Data Dummy (30 fitur, 100 sampel latih, 30 sampel validasi)
np.random.seed(42)
X_train = np.random.rand(100, 30)
y_train = np.random.randint(0, 2, size=(100, 1))

X_val = np.random.rand(30, 30)
y_val = np.random.randint(0, 2, size=(30, 1))

# 2. Membangun Arsitektur Model (5 Neuron per Layer Tersembunyi)
model_5 = Sequential(
    [
        Input(shape=(30,)),
        Dense(5, activation="relu"),  # Diubah dari 10 menjadi 5 neuron
        Dense(5, activation="relu"),  # Diubah dari 10 menjadi 5 neuron
        Dense(1, activation="sigmoid"),
    ]
)

# 3. Kompilasi Model
model_5.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
)

model_5.summary()

# 4. Melatih Model & Mengukur Waktu Pelatihan
start_time = time.time()

history_5 = model_5.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=16,
    validation_data=(X_val, y_val),
    verbose=0,  # Dibuat ringkas agar tidak memenuhi layar
)

training_time_5 = time.time() - start_time

# 5. Evaluasi Hasil Akhir
loss_5, acc_5 = model_5.evaluate(X_val, y_val, verbose=0)

print("\n" + "=" * 45)
print("HASIL EVALUASI MODEL (5 NEURON)")
print("=" * 45)
print(f"Total Parameters : {model_5.count_params()}")
print(f"Training Time    : {training_time_5:.4f} detik")
print(f"Validation Loss  : {loss_5:.4f}")
print(f"Validation Acc   : {acc_5 * 100:.2f}%")
print("=" * 45)