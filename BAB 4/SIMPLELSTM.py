from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential()
# Definisikan input shape secara eksplisit di awal (misal: panjang sequence = 10)
model.add(layers.Input(shape=(10,)))
model.add(layers.Embedding(input_dim=100, output_dim=32))
model.add(layers.LSTM(64))
model.add(layers.Dense(1))
model.summary()