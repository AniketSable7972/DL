"""
Transfer learning on Caltech-101-style folder data:
VGG16 backbone (frozen) → small classifier → optional partial unfreeze + retrain.
"""

# --- Imports: TensorFlow/Keras, VGG16 application, data utilities ---
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# --- Paths & preprocessing: ImageFolder layout under dataset_dir ---
dataset_dir = "caltech-101-img/"
dataset_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
)

# here batch_size is the number of images in each batch
batch_size = 2000
dataset_generator = dataset_datagen.flow_from_directory(
    dataset_dir,
    target_size=(64, 64),
    batch_size=batch_size,
    class_mode='categorical'
)

# First two batches used as train/test tensors (same generator split by index)
x_train, y_train =  dataset_generator[0]
x_test, y_test = dataset_generator[1]

print(len(x_train))
print(len(x_test))

# --- (a) Pre-trained CNN trunk: VGG16 without classification head, 64×64 RGB ---
weights_path = "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
base_model = VGG16(weights=weights_path, include_top=False, input_shape=(64, 64, 3))
# (b) Freezing parameters
for layer in base_model.layers:
   layer.trainable = False

# (c) Adding custom classifier 
x = Flatten()(base_model.output)
x = Dense(64, activation='relu')(x)
predictions = Dense(102, activation='softmax')(x)

# Create the model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

# (d) Training of classifier layers on training data
model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test))

# --- (e) Second stage: new backbone copy, unfreeze top blocks, stronger head + dropout ---
base_model = VGG16(weights=weights_path, include_top=False, input_shape=(64, 64, 3))
# freeze all layers first
for layer in base_model.layers:
   layer.trainable = False
# unfreeze last 4 layers of base model
for layer in base_model.layers[len(base_model.layers) - 2:]:
   layer.trainable = True
# fine-tuning hyper parameters
x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
predictions = Dense(102, activation='softmax')(x)

# Create the model
model = Model(inputs=base_model.input, outputs=predictions)
# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
# training fine tuned model
model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test))

# --- Inspect one test image: softmax argmax vs ground-truth one-hot ---
import matplotlib.pyplot as plt
predicted_value = model.predict(x_test)

labels = list(dataset_generator.class_indices.keys())

n = 1000
plt.imshow(x_test[n])
print("Preditcted: ",labels[np.argmax(predicted_value[n])])
print("Actual: ", labels[np.argmax(y_test[n])])
