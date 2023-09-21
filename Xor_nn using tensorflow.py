import numpy as np
import tensorflow as tf


# XOR gate inputs and outputs
inputs = np.array([[0, 0, 0],
                  [0, 0, 1],
                  [0, 1, 0],
                  [0, 1, 1],
                  [1, 0, 0],
                  [1, 0, 1],
                  [1, 1, 0],
                  [1, 1, 1]])

targets = np.array([[0],
                    [1],
                    [1],
                    [0],
                    [1],
                    [0],
                    [0],
                    [1]])



# Create a sequential model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='sigmoid', input_shape=(3,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics='accuracy')

# Train the model

model.fit(inputs, targets, epochs=10000, verbose=0)


# Test the model
predictions = model.predict(inputs)



print(np.round(predictions))
