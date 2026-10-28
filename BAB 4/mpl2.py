import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, Input
from keras.models import Sequential

# 1. Menyiapkan Data Dummy (Ganti bagian ini jika menggunakan dataset asli)
np.random.seed(42)
X_train = np.random.rand(100, 30)
y_train = np.random.randint(0, 2, size=(100, 1))

X_val = np.random.rand(30, 30)
y_val = np.random.randint(0, 2, size=(30, 1))

# 2. Membangun Model Sequential Modern (Keras 3 Compatible)
model = Sequential(
    [
        Input(shape=(30,)),  # Memakai layer Input eksplisit
        Dense(10, activation="relu"),
        Dense(10, activation="relu"),
        Dense(1, activation="sigmoid"),  # Output biner (0 atau 1)
    ]
)

# 3. Kompilasi Model
model.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
)

# Tampilkan ringkasan arsitektur model
model.summary()

# 4. Melatih Model
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=16,
    validation_data=(X_val, y_val),
    verbose=1,
)

# 5. Evaluasi Hasil Akhir
loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
print("\n" + "=" * 35)
print("Hasil Evaluasi Data Validasi:")
print(f"Loss     : {loss:.4f}")
print(f"Accuracy : {accuracy * 100:.2f}%")
print("=" * 35)

# 6. Plot Grafik Accuracy & Loss (Opsional)
plt.figure(figsize=(12, 4))

# Grafik Loss
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Loss Model")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# Grafik Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.title("Akurasi Model")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.show()