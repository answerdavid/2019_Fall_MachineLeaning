# https://github.com/viraja1/crack_detection/blob/master/models/model.py
# https://subinium.github.io/Keras-3-2/

##################### Use the basic Model #######################
##################### AlexNet ##########################

from __future__ import division, print_function
# coding=utf-8

import os
import numpy as np

# import tensorflow
import tensorflow as tf
import tensorflow.keras as keras

# visualize the model
from IPython.display import SVG
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

# import optimizers from keras
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import backend as K

# import sklearn to use the precision, recall, F1 score
from sklearn import metrics
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score


np.random.seed(1000)
%matplotlib inline


# =============== 1. Define the paths and length, batch size =====================
# os.path.join: create a new path after joining the path
MODEL_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname('content/gdrive')), '/content/gdrive/My Drive/Colab Notebooks/site')
print(MODEL_DATA_PATH)
#train directory
train_dir = os.path.join(MODEL_DATA_PATH, 'train')
#validation directory
validation_dir = os.path.join(MODEL_DATA_PATH, 'test')
# train length
train_length = len(os.listdir(os.path.join(train_dir, '01.temporary(가설)')))+len(os.listdir(os.path.join(train_dir, '02.excavation(토)')))+len(os.listdir(os.path.join(train_dir, '03.foudation(지정)')))+ len(os.listdir(os.path.join(train_dir, '04.rc')))+ len(os.listdir(os.path.join(train_dir, '05.steel(철골)')))+len(os.listdir(os.path.join(train_dir, '06.masonry(조적)')))+len(os.listdir(os.path.join(train_dir, '07.masonry_def')))+len(os.listdir(os.path.join(train_dir, '08.plastering(미장)')))+ len(os.listdir(os.path.join(train_dir, '09.plastering_def')))+ len(os.listdir(os.path.join(train_dir, '10.waterproof(방수)')))+len(os.listdir(os.path.join(train_dir, '11.waterproof_def')))+len(os.listdir(os.path.join(train_dir, '12.carpentry(목공사)')))+ len(os.listdir(os.path.join(train_dir, '13.metal(금속)')))+ len(os.listdir(os.path.join(train_dir, '14.roof(지붕)')))+ len(os.listdir(os.path.join(train_dir, '15.windows(창호,유리)')))+ len(os.listdir(os.path.join(train_dir, '16.windows_def')))+len(os.listdir(os.path.join(train_dir, '17.tiling(타일)')))+len(os.listdir(os.path.join(train_dir, '18.tiling_def')))+len(os.listdir(os.path.join(train_dir, '19.finishing(수장)')))+len(os.listdir(os.path.join(train_dir, '20.finishing_def')))+len(os.listdir(os.path.join(train_dir, '21.landscaping(조경)')))+len(os.listdir(os.path.join(train_dir, '22.painting(도장)')))+len(os.listdir(os.path.join(train_dir, '23.painting_def')))
# validation length
validation_length = len(os.listdir(os.path.join(validation_dir, '01.temporary(가설)')))+len(os.listdir(os.path.join(validation_dir, '02.excavation(토)')))+len(os.listdir(os.path.join(validation_dir, '03.foudation(지정)')))+ len(os.listdir(os.path.join(validation_dir, '04.rc')))+ len(os.listdir(os.path.join(validation_dir, '05.steel(철골)')))+len(os.listdir(os.path.join(validation_dir, '06.masonry(조적)')))+len(os.listdir(os.path.join(validation_dir, '07.masonry_def')))+len(os.listdir(os.path.join(validation_dir, '08.plastering(미장)')))+ len(os.listdir(os.path.join(validation_dir, '09.plastering_def')))+ len(os.listdir(os.path.join(validation_dir, '10.waterproof(방수)')))+len(os.listdir(os.path.join(validation_dir, '11.waterproof_def')))+len(os.listdir(os.path.join(validation_dir, '12.carpentry(목공사)')))+ len(os.listdir(os.path.join(validation_dir, '13.metal(금속)')))+ len(os.listdir(os.path.join(validation_dir, '14.roof(지붕)')))+ len(os.listdir(os.path.join(validation_dir, '15.windows(창호,유리)')))+ len(os.listdir(os.path.join(validation_dir, '16.windows_def')))+len(os.listdir(os.path.join(validation_dir, '17.tiling(타일)')))+len(os.listdir(os.path.join(validation_dir, '18.tiling_def')))+len(os.listdir(os.path.join(validation_dir, '19.finishing(수장)')))+len(os.listdir(os.path.join(validation_dir, '20.finishing_def')))+len(os.listdir(os.path.join(validation_dir, '21.landscaping(조경)')))+len(os.listdir(os.path.join(validation_dir, '22.painting(도장)')))+len(os.listdir(os.path.join(validation_dir, '23.painting_def')))


# batch size
# batch size: 64 -> error
# batch size: 300 -> no error
batch_size = 128
#=================================================================================

# =============== 2. Define the model =====================
# tf.keras.models.Sequential: Lineak stack of layers
model = tf.keras.models.Sequential()

# 1st Convolutional Layer
model.add(Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11),
 strides=(4,4), padding='valid'))
model.add(Activation('relu'))
# Pooling 
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation before passing it to the next layer
model.add(BatchNormalization())

# 2nd Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation
model.add(BatchNormalization())

# 3rd Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Batch Normalisation
model.add(BatchNormalization())

# 4th Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Batch Normalisation
model.add(BatchNormalization())

# 5th Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation
model.add(BatchNormalization())

# Passing it to a dense layer
model.add(Flatten())
# 1st Dense Layer
model.add(Dense(4096, input_shape=(224*224*3,)))
model.add(Activation('relu'))
# Add Dropout to prevent overfitting
model.add(Dropout(0.4))
# Batch Normalisation
model.add(BatchNormalization())

# 2nd Dense Layer
model.add(Dense(4096))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))
# Batch Normalisation
model.add(BatchNormalization())

# 3rd Dense Layer
model.add(Dense(1000))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))
# Batch Normalisation
model.add(BatchNormalization())

# Output Layer
model.add(Dense(23))
model.add(Activation('softmax'))

# print the summary of model
model.summary()
# visualize the model and save as "model_plot.png"
plot_model(model, to_file='model_plot_.png', show_shapes=True, show_layer_names=True)

#=================================================================================

# ================================================================================
def recall(y_target, y_pred):
    # clip(t, clip_value_min, clip_value_max) : clip_value_min~clip_value_max 이외 가장자리를 깎아 낸다
    # round : 반올림한다
    y_target_yn = K.round(K.clip(y_target, 0, 1)) # 실제값을 0(Negative) 또는 1(Positive)로 설정한다
    y_pred_yn = K.round(K.clip(y_pred, 0, 1)) # 예측값을 0(Negative) 또는 1(Positive)로 설정한다

    # True Positive는 실제 값과 예측 값이 모두 1(Positive)인 경우이다
    count_true_positive = K.sum(y_target_yn * y_pred_yn) 

    # (True Positive + False Negative) = 실제 값이 1(Positive) 전체
    count_true_positive_false_negative = K.sum(y_target_yn)

    # Recall =  (True Positive) / (True Positive + False Negative)
    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
    recall = count_true_positive / (count_true_positive_false_negative + K.epsilon())

    # return a single tensor value
    return recall


def precision(y_target, y_pred):
    # clip(t, clip_value_min, clip_value_max) : clip_value_min~clip_value_max 이외 가장자리를 깎아 낸다
    # round : 반올림한다
    y_pred_yn = K.round(K.clip(y_pred, 0, 1)) # 예측값을 0(Negative) 또는 1(Positive)로 설정한다
    y_target_yn = K.round(K.clip(y_target, 0, 1)) # 실제값을 0(Negative) 또는 1(Positive)로 설정한다

    # True Positive는 실제 값과 예측 값이 모두 1(Positive)인 경우이다
    count_true_positive = K.sum(y_target_yn * y_pred_yn) 

    # (True Positive + False Positive) = 예측 값이 1(Positive) 전체
    count_true_positive_false_positive = K.sum(y_pred_yn)

    # Precision = (True Positive) / (True Positive + False Positive)
    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
    precision = count_true_positive / (count_true_positive_false_positive + K.epsilon())

    # return a single tensor value
    return precision


def f1score(y_target, y_pred):
    _recall = recall(y_target, y_pred)
    _precision = precision(y_target, y_pred)
    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
    _f1score = ( 2 * _recall * _precision) / (_recall + _precision+ K.epsilon())
    
    # return a single tensor value
    return _f1score

# =================================================================================

# =============== 3. Set the training model =====================
# set the training
# loss: binary_crossentropy(or categorical_crossentropy, mse)
# optimizer: adam, sgd
# metrics: use to monitor the training
#opt = optimizers.Adam(lr = 0.001)
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(lr = 0.01),
              metrics=['acc',
                       precision,
                       recall,
                       f1score
                       #single_class_precision(0), single_class_recall(0),
                       ])
#=================================================================================



# =============== 4. Creat the data =====================
# generate batches of tensor image data with real-time data augmentation
# data argumentation
# featurewise_std_normalization: divide inputs by std of the dataset, feature-wise
# shear_range: shear intensity
# zoom range: range for random zoom
# rotation range: defree range for random rotations
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255,
                                                                featurewise_std_normalization = True,
                                                                shear_range= 0.2,
                                                                zoom_range = 0.2,
                                                                rotation_range = 90)
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
                                                                #featurewise_std_normalization = True,
                                                                #shear_range = 0.2,
                                                                #zoom_range = 0.2,
                                                                #rotation_range = 90)

# load the data set
# 1st argument: the path of image
# target_size: width and length of image(128*128)
# batch_size = 64
# class_mode: binary(return a 1D binary label)
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(224, 224), 
                                                    batch_size=batch_size, 
                                                    class_mode='categorical')

validation_generator = train_datagen.flow_from_directory(validation_dir,
                                                         target_size=(224, 224), 
                                                         batch_size=batch_size,
                                                         class_mode='categorical')
#=================================================================================




# =============== 5. Train the model =====================
# train the model
# model.fit_generator: train the batch produced by generator
history = model.fit_generator(
      train_generator,
      # define the number of step in 1 epoch
      steps_per_epoch=int(train_length / batch_size),
      # iteration number
      epochs=50,
      verbose=1,
      # validation data set
      validation_data=validation_generator,
      # the number of step in validation per 1 epoch
      validation_steps=3
)
print(train_length / batch_size)
#=================================================================================



# =============== 6. Evaluate the model =====================
# evaluate the model
print("--- Evaluate ---")

scores = model.evaluate_generator(validation_generator, steps = 3)
print("%s: %.2f%%" % (model.metrics_names[1],scores[1]*100))
#=================================================================================



# =============== 7.Predict the model =====================
#predict the model
print("--- Predict ---")

#output
output = model.predict_generator(validation_generator, steps = 3.04)
val_preds = np.argmax(output, axis = -1)
val_trues = validation_generator.classes

print('\n------- val_trues ------\n')
print(val_trues)
cm = metrics.confusion_matrix(val_trues, val_preds)
print(cm)
np.set_printoptions(formatter={'float':lambda x: "{0:0.3f}".format(x)})
print(validation_generator.class_indices)
print('\n------- output ------\n')
print(output)

# print out the confusion matrix
#print('\nConfusion Matrix')
#print(len(validation_generator.classes))
#print("length of val_preds: ", len(val_preds))
#print("length of output: ",len(output))
#print("\n value preds \n", val_preds)
#print(confusion_matrix(output, val_preds))
#print('\nClassification Report')
#target_names = ['0', '1', '2', '3', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
#print(classification_report(validation_generator.classes, y_pred, target_names=target_names))
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
plt.show()

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()


#=================================================================================

#model.save(os.path.join(os.path.dirname(__file__), 'crack_detection.h5'))