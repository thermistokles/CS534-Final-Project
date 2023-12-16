import argparse
from pathlib import Path
import datetime
import time
import json
import numpy as np 
import pandas as pd 
import torch
import tensorflow as tf
import torch_tensorrt
import keras
import matplotlib.pyplot as plt
import os
import sys
sys.path.append("D:\Anaconda\Lib\site-packages")
import cv2
import repvit
from timm.models import create_model
from main import main
from main import get_args_parser
import utils
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

model=create_model('repvit_m0_9')
utils.replace_batchnorm(model)
if __name__=='__main__':
    parser = get_args_parser()
    args=parser.parse_args()
    if args.resume and not args.eval:
       	args.output_dir = '/'.join(args.resume.split('/')[:-1])
    elif args.output_dir:
       	args.output_dir = args.output_dir + f"/{args.model}/" + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    else:
       	assert(False)
    args.model=model
    args.data_path='~/imagenet'
    args.batch_size=32
    args.epochs=20
    args.model=model
    args.dist_eval=False
    main(args)