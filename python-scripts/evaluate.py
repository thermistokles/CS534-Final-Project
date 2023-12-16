import numpy as np 
import pandas as pd 
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import os
import cv2

from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from keras.applications.convnext import LayerScale

load_model = tf.keras.models.load_model

MODEL_NAME = "results-292410/cvnext-tiny.keras"
# MODEL_NAME = "results-postaug/aug-cvntiny.keras"
model = load_model(MODEL_NAME,custom_objects={
    "LayerScale":LayerScale
})

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
predictions = model.predict(npImageList, verbose=1)
predictions = np.argmax(predictions, axis=1)
ground_truths = imageData.label
print(f"MODEL EVALUATION FOR {MODEL_NAME}")

print("\nAccuracy: "+str(test_acc)+", Loss: "+str(test_loss))

 # Confusion matrix
cm = confusion_matrix(ground_truths, predictions, normalize="all")
print("Confusion Matrix:")
print(str(cm))

print("F1 Score: %.4f" % (f1_score(ground_truths, predictions)))

print("MCC Score: %.4f" % (matthews_corrcoef(ground_truths, predictions)))
