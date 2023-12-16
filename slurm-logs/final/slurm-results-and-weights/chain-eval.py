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

MODEL_TINY = "results-292410/cvnext-tiny.keras"
mt = load_model(MODEL_TINY,custom_objects={
    "LayerScale":LayerScale
})

MODEL_SMALL = "results-292410/cvnext-small.keras"
ms = load_model(MODEL_TINY,custom_objects={
    "LayerScale":LayerScale
})

MODEL_BASE = "results-292410/cvnext-base.keras"
mb = load_model(MODEL_BASE,custom_objects={
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

print(f"--------------------------------\n{MODEL_TINY}\n--------------------------------\n")

test_loss,test_acc = mt.evaluate(npImageList,ytest,verbose=1)
predictions = mt.predict(npImageList, verbose=1)
predictions = np.argmax(predictions, axis=1)
ground_truths = imageData.label
print("\nAccuracy: "+str(test_acc)+", Loss: "+str(test_loss))

 # Confusion matrix
cm = confusion_matrix(ground_truths, predictions, normalize="all")
print("Confusion Matrix:")
print(str(cm))

print("F1 Score: %.4f" % (f1_score(ground_truths, predictions)))

print("MCC Score: %.4f\n" % (matthews_corrcoef(ground_truths, predictions)))

print(f"--------------------------------\n{MODEL_SMALL}\n--------------------------------\n")

test_loss,test_acc = ms.evaluate(npImageList,ytest,verbose=1)
predictions = ms.predict(npImageList, verbose=1)
predictions = np.argmax(predictions, axis=1)
ground_truths = imageData.label
print("\nAccuracy: "+str(test_acc)+", Loss: "+str(test_loss))

 # Confusion matrix
cm = confusion_matrix(ground_truths, predictions, normalize="all")
print("Confusion Matrix:")
print(str(cm))

print("F1 Score: %.4f" % (f1_score(ground_truths, predictions)))

print("MCC Score: %.4f\n" % (matthews_corrcoef(ground_truths, predictions)))

print(f"--------------------------------\n{MODEL_BASE}\n--------------------------------\n")

test_loss,test_acc = mb.evaluate(npImageList,ytest,verbose=1)
predictions = mb.predict(npImageList, verbose=1)
predictions = np.argmax(predictions, axis=1)
ground_truths = imageData.label
print("\nAccuracy: "+str(test_acc)+", Loss: "+str(test_loss))

 # Confusion matrix
cm = confusion_matrix(ground_truths, predictions, normalize="all")
print("Confusion Matrix:")
print(str(cm))

print("F1 Score: %.4f" % (f1_score(ground_truths, predictions)))

print("MCC Score: %.4f\n" % (matthews_corrcoef(ground_truths, predictions)))
