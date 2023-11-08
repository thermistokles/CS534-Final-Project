import numpy as np 
import pandas as pd 
import os
import albumentations as A
import cv2
import random

imageData=pd.read_csv('stage_1_train_images.csv')

images='png_images'
masks='png_masks'
# label = has_pneumo
imageData['label']=imageData['has_pneumo']
imageData['path']=imageData['new_filename'].apply(lambda x:os.path.join(images,x))
imageData['mpath']=imageData['new_filename'].apply(lambda x:os.path.join(masks,x))

il_pos = []
il_neg = []
for i in range(len(imageData)):
    image = imageData.path.iloc[i]
    if(imageData.label.iloc[i]==1):
        il_pos.append(cv2.resize(cv2.imread(image,cv2.IMREAD_COLOR),[224,224]))
    else:
        il_neg.append(cv2.resize(cv2.imread(image,cv2.IMREAD_COLOR),[224,224]))

npILpos = np.array(il_pos)
npILneg = np.array(il_neg)

transform = A.Compose([
    A.Rotate(limit=20, p=1),
    A.ShiftScaleRotate(shift_limit=0, scale_limit=0.15, rotate_limit=0, p=1),
])

for i in range(100):
    pos_img = random.choice(il_pos)
    neg_img = random.choice(il_neg)
    pos_img_transformed = transform(image=pos_img)["image"]
    neg_img_transformed = transform(image=neg_img)["image"]
    pos_filename = f"posadjusted-{i:05d}.png"
    neg_filename = f"negadjusted-{i:05d}.png"
    cv2.imwrite(pos_filename,pos_img_transformed)
    cv2.imwrite(neg_filename,neg_img_transformed)
    