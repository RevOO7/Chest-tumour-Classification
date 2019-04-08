#Resize
import cv2
import pandas as pd
import os
import numpy as np
from tqdm import tqdm

train = pd.read_csv(os.path.join(os.getcwd(), 'train.csv'))
test = pd.read_csv(os.path.join(os.getcwd(), 'test.csv'))

TRAIN_PATH = os.path.join(os.getcwd(), 'train_/')
TEST_PATH = os.path.join(os.getcwd(), 'x/')


def read_img(img_path):
    img = cv2.imread(img_path, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(128,128))
    return img


for img_path in tqdm(train['image_name'].values):
    cv2.imwrite('train_pp/'+img_path, read_img(TRAIN_PATH + img_path))
    
for img_path in tqdm(test['image_name'].values):
    cv2.imwrite('test_pp/'+img_path, read_img(TEST_PATH + img_path))
    
del img_path

# =============================================================================
# 
# =============================================================================

#JPEG TO PNG CONVERTER
import os
from PIL import Image

target_directory = '.'
target = '.png'

for file in os.listdir(target_directory):
    filename, extension = os.path.splitext(file)
    img = Image.open(filename + extension)
    img.save('../x/'+filename + target)

