
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, Dropout, MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt

# --- Load dataset: 70k grayscale 28×28 images of digits 0–9 ---
mnist = tf.keras.datasets.mnist
(x_train,y_train), (x_test,y_test) = mnist.load_data()

# --- Preprocess: add channel dim (N, H, W) → (N, H, W, 1) for Conv2D ---
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
print("Data type of x_train", x_train.dtype)

# --- Cast to float32 (required for neural net math / GPU) ---
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
print("Data type of x_train", x_train.dtype)

# --- Model: Conv → pool → flatten → dense → dropout → 10-class softmax ---
model = Sequential()
model.add(Conv2D(28,kernel_size=(3,3), input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(100,activation="relu"))
model.add(Dropout(0,3))
model.add(Dense(10,activation="softmax"))

model.summary()

# Train Model
model.compile( 
    optimizer="adam", 
    loss="sparse_categorical_crossentropy", 
    metrics=["accuracy"]
)
model.fit(x_train,y_train,epochs=10)

# Estimation Model performance 
test_loss, test_acc = model.evaluate(x_test,y_test)
print("Loss: ",test_loss)
print("Accuracy: ",test_acc)

# --- Spot-check: random training image, visualize, then predict class ---
import random 
n = random.randint(0,9999)
img = x_train[n]
plt.imshow(-np.squeeze(img), cmap='gray')
plt.show()
image = img.reshape(1,img.shape[0],img.shape[1],img.shape[2])
predict_model = model.predict([image])
print("Predicted class: {}".format(np.argmax(predict_model)))
