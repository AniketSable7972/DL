# Importing required libraries (note: formatting issues exist but kept as-is)
import tensorflow as tffrom tensorflow 
import kerasfrom tensorflow.keras.preprocessing.image 
import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

# Loading MNIST dataset (handwritten digits)
mnist = tf.keras.datasets.mnist(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Checking number of samples in dataset
len(x_train)   # Training images count
len(x_test)    # Testing images count
len(y_train)   # Training labels count
len(y_test)    # Testing labels count

# Inspecting shape and sample data
x_train.shape  # Shape of training dataset
x_train[0]     # First image pixel values

# Visualizing first image
plt.matshow(x_train[0])  # Matrix view of image
plt.imshow(-x_train[0], cmap="gray")  # Inverted grayscale view

# Normalizing pixel values (0–255 → 0–1)
x_train = x_train / 255x_test = x_test /255

# Checking normalized image
x_train[0]

# Building neural network model
model = keras.Sequential([ 
    keras.layers.Flatten(input_shape=(28, 28, 1)),  # INPUT LAYER: converts 2D image to 1D vector
    keras.layers.Dense(128, activation="relu"),     # HIDDEN LAYER: learns patterns
    keras.layers.Dense(10, activation="softmax")    # OUTPUT LAYER: probability for digits (0–9)
])

# Display model structure
model.summary()

# Compiling model (define optimizer, loss function, and metrics)
model.compile( 
    optimizer="sgd",  # Optimization algorithm
    loss="sparse_categorical_crossentropy",  # Loss function for classification
    metrics=["accuracy"  # Accuracy metric
])

# Training the model
history = model.fit(x_train, y_train, epochs=8, validation_data=(x_test, y_test))

# Evaluating model on test data
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Loss= ", test_loss)
print("Accuracy ", test_acc)

# Displaying a test image
n = 20
plt.imshow(x_test[n])
plt.show()

# Making predictions on test data
# WE USE PREDICT() ON NEW DATA
predicted_value = model.predict(x_test)

# Comparing actual vs predicted value
print("Actual Number: ", np.argmax(y_test[n]))
print("Predicted Number: ", np.argmax(predicted_value[n]))

# Extracting training history
history = history.history
history.keys()

# Keys present in history dictionary
dict_keys(['accuracy', 'loss', 'val_accuracy', 'val_loss'])

# Plotting training vs validation accuracy
plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title("Model Accuracy")
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', "Validation"], loc='upper left')
plt.show()

# Plotting training vs validation loss
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Combined plot for accuracy and loss
plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('Training Loss and accuracy')
plt.ylabel('accuracy/Loss')
plt.xlabel('epoch')
plt.legend(['accuracy', 'val_accuracy', 'loss', 'val_loss'])