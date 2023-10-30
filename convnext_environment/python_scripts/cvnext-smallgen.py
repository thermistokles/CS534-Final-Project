import numpy as np 
import pandas as pd 
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import os
import cv2

from keras.preprocessing.image import ImageDataGenerator
load_model = tf.keras.models.load_model

mfiles=[]
mpaths = []
for dirname, _, filenames in os.walk('png_masks'):
    for filename in filenames:
        path = os.path.join(dirname, filename)    
        mpaths.append(path)
        mfiles.append(filename)

files=[]
paths = []
for dirname, _, filenames in os.walk('png_images'):
    for filename in filenames:
        path = os.path.join(dirname, filename)    
        paths.append(path)
        files.append(filename)

imageData=pd.read_csv('stage_1_train_images.csv')

images='png_images'
masks='png_masks'
# label = has_pneumo
imageData['label']=imageData['has_pneumo']
imageData['path']=imageData['new_filename'].apply(lambda x:os.path.join(images,x))
imageData['mpath']=imageData['new_filename'].apply(lambda x:os.path.join(masks,x))

imageList = []
for image in imageData.path:
    imageList.append(cv2.resize(cv2.imread(image,cv2.IMREAD_COLOR),[224,224]))

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

ytrain = keras.utils.to_categorical(imageData.label)

indicies = np.random.permutation(len(imageList))
indicies_training = indicies[:8540]
indicies_validation = indicies[8540:]

trainingData = npImageList[indicies_training,:,:,:]
validationData = npImageList[indicies_validation,:,:,:]

label_training = ytrain[indicies_training,:]
label_validation = ytrain[indicies_validation,:]

model.compile(optimizer="Adam",loss="categorical_crossentropy",metrics=["acc"])
early_stopping = keras.callbacks.EarlyStopping( 
    monitor='val_loss', 
    patience=10 
)
lr_reduction = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=.5, 
    patience=3, 
    verbose=1 
)
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15, fill_mode="nearest")

model.summary()
model.fit(
    aug.flow(trainingData,
    label_training,batch_size=64),
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
MODEL_NAME = "convnext_small-augmented.keras"
model.save(MODEL_NAME)

model = load_model(MODEL_NAME)

imageData=pd.read_csv('stage_1_test_images.csv')

images='png_images'
masks='png_masks'
# label = has_pneumo
imageData['label']=imageData['has_pneumo']
imageData['path']=imageData['new_filename'].apply(lambda x:os.path.join(images,x))
imageData['mpath']=imageData['new_filename'].apply(lambda x:os.path.join(masks,x))

imageList = []
for image in imageData.path:
    imageList.append(cv2.resize(cv2.imread(image,cv2.IMREAD_COLOR),[224,224]))

npImageList = np.array(imageList)
ytest = keras.utils.to_categorical(imageData.label)

test_loss,test_acc = model.evaluate(npImageList,ytest,verbose=1)
print("Accuracy: "+str(test_acc)+", Loss: "+str(test_loss))
