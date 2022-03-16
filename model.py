# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 09:58:51 2020

@author: Sheeza
"""

import numpy as np
from keras import backend as K
from keras.optimizers import Adam, SGD

from keras.layers import Flatten, Dropout, BatchNormalization, Reshape, GlobalAveragePooling2D
from keras.layers import Dense, Dropout, Input
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler

from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import pandas as pd

from keras.preprocessing import image

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import matplotlib.image as mpimg

import shutil
import os

images_folder = os.listdir("images")
first_folder = images_folder[0]
benign_images_first_folder = f'images/{first_folder}/0'
benign_images = os.listdir(benign_images_first_folder)[0:3]
malignant_images_first_folder = f'images/{first_folder}/1'
malignant_images = os.listdir(malignant_images_first_folder)[0:3]
def plot_images(image_index, folder_name, image_name, title):
  sp = figure.add_subplot(2, 3, image_index + 1)
  sp.axis('Off')
  image_path = f'{folder_name}/{image_name}'
  image_loaded =mpimg.imread(image_path)
  sp.set_title(title, fontsize=16)
  plt.imshow(image_loaded, interpolation=None)

figure = plt.figure(figsize=(12, 6))

for image_index, image_name in enumerate(benign_images):
  plot_images(image_index, benign_images_first_folder, image_name, "Benign")


for image_index, image_name in enumerate(malignant_images):
  plot_images(image_index + 3, malignant_images_first_folder, image_name, "Malignant")

#par_dir_T=r"C:\Users\Sheeza\Desktop\breast_cancer\Breast-cancer-imges\training"
#os.mkdir('training') 
#os.mkdir('validation') 
#path_benign_T = os.path.join(par_dir_T, 'benign')
#os.mkdir(path_benign_T)
#path_Malignant_T = os.path.join(par_dir_T, 'Malignant')
#os.mkdir(path_Malignant_T)


#par_dir_V=r"C:\Users\Sheeza\Desktop\breast_cancer\Breast-cancer-imges\validation"
#path_benign_V = os.path.join(par_dir_V, 'benign')
#os.mkdir(path_benign_V)
#path_Malignant_V= os.path.join(par_dir_V, 'Malignant')
#os.mkdir(path_Malignant_V)

benign_images_training_folder_name = "training/benign"
malignant_images_training_folder_name = "training/malignant"
for folder in images_folder:
  benign_folder = os.listdir(f'images/{folder}/0')
  malignant_folder = os.listdir(f'images/{folder}/1')
  
  for benign_image in benign_folder:
    image_url = f'images/{folder}/0/{benign_image}'
    shutil.move(image_url, benign_images_training_folder_name)
    
  for malign_image in malignant_folder:
    image_url = f'images/{folder}/1/{malign_image}'
    shutil.move(image_url, malignant_images_training_folder_name)


benign_images = len(os.listdir(benign_images_training_folder_name))
malignant_images = len(os.listdir(malignant_images_training_folder_name))
total_images =  benign_images + malignant_images

print(f'Total images: {total_images}')
print(f'Benign images: {benign_images}')
print(f'Malignant images: {malignant_images}')


validation_size = 0.20

benign_validation_folder_size = int(benign_images * 0.20)
malignant_validation_folder_size = int(malignant_images * 0.20)
print(f'Benign validation size: {benign_validation_folder_size}')
print(f'Malignant validation size: {malignant_validation_folder_size}')

benign_images_validation_folder_name = "validation/benign"
malignant_images_validation_folder_name = "validation/malignant"

benign_images_training_folder = os.listdir(benign_images_training_folder_name)
malignant_images_training_folder = os.listdir(malignant_images_training_folder_name)

for image_name in benign_images_training_folder[:benign_validation_folder_size]:
  image_url = f'{benign_images_training_folder_name}/{image_name}'
  shutil.move(image_url, benign_images_validation_folder_name)
  
for image_name in malignant_images_training_folder[:malignant_validation_folder_size]:
  image_url = f'{malignant_images_training_folder_name}/{image_name}'
  shutil.move(image_url, malignant_images_validation_folder_name)
  
training_folder_size = len(os.listdir(benign_images_training_folder_name)) + len(os.listdir(malignant_images_training_folder_name))
validation_folder_size = len(os.listdir(benign_images_validation_folder_name)) + len(os.listdir(malignant_images_validation_folder_name))  

print(f'Training folder size: {training_folder_size}')
print(f'Validation folder size: {validation_folder_size}')



batch_size = 128
img_size = 48
input_img_size = (48, 48, 3)
num_classes = 2

train_generator = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    zoom_range=0.05,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    horizontal_flip=True,
    fill_mode="nearest").flow_from_directory(
    "training",
    target_size=(img_size, img_size),
    color_mode="rgb",
	  shuffle=True,
    batch_size=batch_size)

val_generator = ImageDataGenerator(
    preprocessing_function=preprocess_input).flow_from_directory(
    "validation",
    target_size=(img_size, img_size),
    color_mode="rgb",
	  shuffle=False,
    batch_size=batch_size)

train_steps = int(training_folder_size // batch_size)
val_steps = int(validation_folder_size // batch_size) 

train_steps, val_steps

def create_model():
  input_tensor = Input(shape=input_img_size)
  
  mobile_model = MobileNetV2(
       weights=None,
       input_tensor=input_tensor,
       input_shape=input_img_size,
       alpha=1.5,
       include_top=False)
  
  for layer in mobile_model.layers:
    layer.trainable = True
  
  mobile_model_output = mobile_model.output
  classification_layer = Flatten()(mobile_model_output)
  classification_layer = Dense(256, activation='relu')(classification_layer)
  classification_layer = Dropout(0.5)(classification_layer)
  predictions = Dense(activation="softmax", units=num_classes)(classification_layer)

  model = Model(inputs=input_tensor, outputs=predictions)
  
  return model

learning_rate = 3e-4
epochs = 20

def polynomial_decay(epoch):
	power = 1.0
 
	alpha = learning_rate * (1 - (epoch / float(epochs))) ** power
	return alpha

y_true = np.concatenate((val_generator.classes, train_generator.classes))
cw = compute_class_weight('balanced', np.unique(y_true), y_true)
class_weights = {index: weight for index, weight in enumerate(cw)}
weights_name = "epoch={epoch:02d}|accuracy={val_acc:.4f}.h5"

checkpoint = ModelCheckpoint(weights_name, monitor="val_acc", verbose=1, save_best_only=True,
                                 save_weights_only=True, mode="max", period=1)

lr_decay = LearningRateScheduler(polynomial_decay)

optimizer = SGD(lr=learning_rate, momentum=0.9)

model = create_model()
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc'])
trained_model = model.fit_generator(train_generator,
                        epochs=epochs,
                        class_weight=class_weights,
                        steps_per_epoch=train_steps,
                        callbacks=[checkpoint, lr_decay],
                        validation_data=val_generator,
                        validation_steps=val_steps,
                        verbose=1)

def plot_validation_training(metric, trained_model):
  validation_metric = trained_model.history[f'val_{metric}']
  training_metric = trained_model.history[metric]
  epochs = range(len(training_metric))
  plt.plot(epochs, training_metric, 'b', label=f'Training {metric}')
  plt.plot(epochs, validation_metric, 'r', label=f'Validation {metric}')
  plt.ylim(bottom=0)
  plt.xlabel('Epochs ', fontsize=16)
  plt.ylabel(metric, fontsize=16)
  loc = 'upper right' if metric == "loss" else 'lower right'
  plt.legend(loc=loc)
  plt.title(f'Training and validation {metric}', fontsize = 20)
  plt.show()
  
plot_validation_training("loss", trained_model)

plot_validation_training("acc", trained_model)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
model.load_weights('epoch=15_accuracy=0.8687.h5')
val_generator.reset()
classes_predicted = model.predict_generator(val_generator, steps=val_steps, verbose=1)
len(classes_predicted)
real_classes = np.argmax(classes_predicted, axis=1)
val_labels = val_generator.classes
len(val_labels)
val_labels = val_labels[:55424]
len(val_labels), len(real_classes)
val_generator.class_indices
classes_names = ["Benign", "Malignant"]
cm = confusion_matrix(val_labels, real_classes, labels=range(num_classes))
plot_confusion_matrix(cm, classes_names)
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
print(f'sensitivity: {sensitivity}')

specificity = cm[1, 1] / (cm[1, 1] + cm[1, 0])
print(f'specifity: {specificity}')
from sklearn.metrics import classification_report
report = classification_report(val_labels, real_classes, target_names=classes_names)
print(report)



model_json = model.to_json()
with open("breast_cancer_model.json", "w") as json_file:
    json_file.write(model_json)





