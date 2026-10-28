from keras.layers import Activation, Dense, Input, SimpleRNN
from keras.models import Sequential
from keras import optimizers

model = Sequential()
model.add(Input(shape=(49, 1)))
model.add(SimpleRNN(50, return_sequences=False))
model.add(Dense(46))
model.add(Activation("softmax"))

# Ubah 'lr' menjadi 'learning_rate'
adam = optimizers.Adam(learning_rate=0.001)

model.compile(
    loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"]
)
print(model.summary())