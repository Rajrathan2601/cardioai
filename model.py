import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input
from tensorflow.keras.models import Model

# Dummy ECG data
X = np.random.rand(400,200,1)
y = tf.keras.utils.to_categorical(np.random.randint(0,5,400))

inputs = Input(shape=(200,1))

x = Conv1D(32,3,activation='relu')(inputs)
x = MaxPooling1D(2)(x)

# IMPORTANT LAYER
x = Conv1D(64,3,activation='relu', name="last_conv")(x)

x = MaxPooling1D(2)(x)
x = Flatten()(x)
x = Dense(64,activation='relu')(x)

outputs = Dense(5,activation='softmax')(x)

model = Model(inputs, outputs)

model.compile(optimizer='adam', loss='categorical_crossentropy')

model.fit(X,y,epochs=3)

model.save("ecg_model.h5")

print("✅ Model ready!")