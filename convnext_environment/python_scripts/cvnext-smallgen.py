import numpy as np 
import pandas as pd 
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import os
import cv2
import sys
import datetime

from keras.optimizers import SGD

print(f"SYSTEM TIME AT EXECUTION: {datetime.datetime.now()}")

imageList = []
labels = []
for i in range(20000):
    pos_filename = f"augmentation/posadjusted-{i:05d}.png"
    neg_filename = f"augmentation/negadjusted-{i:05d}.png"
    imageList.append(cv2.imread(pos_filename,cv2.IMREAD_COLOR))
    labels.append(1)
    imageList.append(cv2.imread(neg_filename,cv2.IMREAD_COLOR))
    labels.append(0)

npImageList = np.array(imageList)

model = tf.keras.applications.ConvNeXtSmall(
    model_name="convnext_small",
    include_top=True,
    include_preprocessing=True,
    weights=None,
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=2,
    classifier_activation="softmax",
)

ytrain = keras.utils.to_categorical(labels)

indicies = np.random.permutation(len(imageList))
indicies_training = indicies[:32000]
indicies_validation = indicies[32000:]

trainingData = npImageList[indicies_training,:,:,:]
validationData = npImageList[indicies_validation,:,:,:]

label_training = ytrain[indicies_training,:]
label_validation = ytrain[indicies_validation,:]

## model.compile(optimizer="Adam",loss="categorical_crossentropy",metrics=["acc"])
model.compile(optimizer=SGD(lr=0.01),loss="categorical_crossentropy",metrics=["acc"])
early_stopping = keras.callbacks.EarlyStopping( 
    monitor='val_loss', 
    patience=50 
)
lr_reduction = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=.5,
    patience=3,
    verbose=1 
)

model.summary()
model.fit(
    trainingData,
    label_training,
    batch_size=32,
    verbose=1,
    epochs=10000,
    validation_data=[
        validationData,
        label_validation
    ],
    callbacks=[
        early_stopping,
        lr_reduction
    ]
)
MODEL_NAME = "cvnext-small.keras"
model.save(MODEL_NAME)