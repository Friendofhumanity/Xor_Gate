# importing libraries
import numpy as np
import tensorflow as tf


# XOR gate inputs , Can be modified according to your requirements
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



# Creating a sequential model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='sigmoid', input_shape=(3,)),   # I have used 3 inputs layers 4 hidden layers and
                                                                 
    tf.keras.layers.Dense(1, activation='sigmoid')                       # 1 output layer
])

# Compile the model 

model.compile(optimizer='adam', loss='mean_squared_error', metrics='accuracy') 

# Train the model

model.fit(inputs, targets, epochs=10000, verbose=0)  #you can set the verbose as per your requirement 


# Test the model
predictions = model.predict(inputs)


# print the predictions
print(np.round(predictions))
