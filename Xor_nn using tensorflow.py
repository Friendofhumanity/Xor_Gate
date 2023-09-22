# importing libraries
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
# Trarget for the purpose of training 
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
    tf.keras.layers.Dense(4, activation='sigmoid', input_shape=(3,)),   # We haev used 3 inputs layers 4 hidden layers and
                                                                         # 1 output layer 
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics='accuracy')

# Train the model

model.fit(inputs, targets, epochs=10000, verbose=0)  #you can set the verbose as per your requirement 


# Test the model
predictions = model.predict(inputs)



print(np.round(predictions))
