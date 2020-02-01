# https://github.com/viraja1/crack_detection/blob/master/models/model.py
# https://subinium.github.io/Keras-3-2/

##################### Use the basic Model #######################


from __future__ import division, print_function
# coding=utf-8

import os
import numpy as np

# import tensorflow
import tensorflow as tf

# visualize the model
from IPython.display import SVG
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

%matplotlib inline


# =============== 1. Define the paths and length, batch size =====================
# os.path.join: create a new path after joining the path
MODEL_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname('content/gdrive')), '/content/gdrive/My Drive/Colab Notebooks/data/crack_dataset.tar')
print(MODEL_DATA_PATH)
#train directory
train_dir = os.path.join(MODEL_DATA_PATH, 'train')
#validation directory
validation_dir = os.path.join(MODEL_DATA_PATH, 'validation')
# train length
train_length = len(os.listdir(os.path.join(train_dir, 'crack'))) + len(os.listdir(os.path.join(train_dir, 'no_crack')))
# validation length
validation_length = len(os.listdir(os.path.join(validation_dir, 'crack'))) + \
                    len(os.listdir(os.path.join(validation_dir, 'no_crack')))
# batch size
batch_size = 300
#=================================================================================

# =============== 2. Define the model =====================
# tf.keras.models.Sequential: Lineak stack of layers
model = tf.keras.models.Sequential([
 # 2D convolution layer, activation function: relu
 # 128 * 128 RGB pictures
 # number of filterts = 32
 # kernel_size = (3,3)
 tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
 # max pooling, pool_size = (2,2)
 tf.keras.layers.MaxPool2D((2, 2)),
 tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
 tf.keras.layers.MaxPool2D((2, 2)),
 tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
 tf.keras.layers.MaxPool2D((2, 2)),
 # flatten the data, does not affect the batch size
 tf.keras.layers.Flatten(),
 # fully connected layer
 tf.keras.layers.Dense(units=512, activation='relu'),
 tf.keras.layers.Dense(units=1, activation='sigmoid'),
])
# print the summary of model
model.summary()
# visualize the model and save as "model_plot.png"
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

#=================================================================================



# =============== 3. Set the training model =====================
# set the training
# loss: binary_crossentropy(or categorical_crossentropy, mse)
# optimizer: adam, sgd
# metrics: use to monitor the training
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])
#=================================================================================



# =============== 4. Creat the data =====================
# generate batches of tensor image data with real-time data augmentation
# data argumentation
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)

# load the data set
# 1st argument: the path of image
# target_size: width and length of image(128*128)
# batch_size = 300
# class_mode: binary(return a 1D binary label)
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(128, 128), batch_size=batch_size, class_mode='binary')

validation_generator = train_datagen.flow_from_directory(validation_dir,
                                                         target_size=(128, 128), batch_size=batch_size,
                                                         class_mode='binary')
#=================================================================================




# =============== 5. Train the model =====================
# train the model
# model.fit_generator: train the batch produced by generator
history = model.fit_generator(
      train_generator,
      # define the number of step in 1 epoch
      steps_per_epoch=int(train_length / batch_size),
      # iteration number
      epochs=15,
      verbose=1,
      # validation data set
      validation_data=validation_generator,
      # the number of step in validation per 1 epoch
      validation_steps=5
)
#=================================================================================



# =============== 6. Evaluate the model =====================
# evaluate the model
print("--- Evaluate ---")

scores = model.evaluate_generator(validation_generator, steps = 5)
print("%s: %.2f%%" % (model.metrics_names[1],scores[1]*100))
#=================================================================================



# =============== 7.Predict the model =====================
#predict the model
print("--- Predict ---")

output = model.predict_generator(validation_generator, steps = 5)
np.set_printoptions(formatter={'float':lambda x: "{0:0.3f}".format(x)})
print(validation_generator.class_indices)
print(output)
#=================================================================================


# =============== 8.Draw the result =====================
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(acc)+1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

#=================================================================================

#model.save(os.path.join(os.path.dirname(__file__), 'crack_detection.h5'))