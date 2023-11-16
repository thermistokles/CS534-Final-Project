import numpy as np 
import pandas as pd 
import os
import albumentations as A
import cv2
import random
import sys

print("Reading images from original training set serialization")
imageData=pd.read_csv('stage_1_train_images.csv')

images='png_images'
masks='png_masks'
# label = has_pneumo
imageData['label']=imageData['has_pneumo']
imageData['path']=imageData['new_filename'].apply(lambda x:os.path.join(images,x))
imageData['mpath']=imageData['new_filename'].apply(lambda x:os.path.join(masks,x))

il_pos = []
il_neg = []
print("Preprocessing images from original training set")
for i in range(len(imageData)):
    if i%100==0:
        print(f"Iteration {i} in progress...   ",end=" ")
        sys.stdout.flush()
    image = imageData.path.iloc[i]
    if(imageData.label.iloc[i]==1):
        il_pos.append(cv2.resize(cv2.imread(image,cv2.IMREAD_COLOR),[512,512]))
    else:
        il_neg.append(cv2.resize(cv2.imread(image,cv2.IMREAD_COLOR),[512,512]))

transform = A.Compose([
    A.Rotate(limit=10, p=1),
    A.ShiftScaleRotate(shift_limit=0, scale_limit=0.05, rotate_limit=0, p=1),
])

print("Running image transforms")
for i in range(20000):
    if i%500==0:
        print(f"Iteration {i} in progress...   ",end=" ")
        sys.stdout.flush()
    pos_img = random.choice(il_pos)
    neg_img = random.choice(il_neg)
    pos_img_transformed = cv2.resize(transform(image=pos_img)["image"],[224,224])
    neg_img_transformed = cv2.resize(transform(image=neg_img)["image"],[224,224])
    pos_filename = f"aug2/posadjusted-{i:05d}.png"
    neg_filename = f"aug2/negadjusted-{i:05d}.png"
    cv2.imwrite(pos_filename,pos_img_transformed)
    cv2.imwrite(neg_filename,neg_img_transformed)
print("")
print("Image transformation completed")
    