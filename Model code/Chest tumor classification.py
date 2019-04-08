import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Convolution2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from scipy import misc
import cv2

train = pd.read_csv(os.path.join(os.getcwd(), 'train.csv'))
test = pd.read_csv(os.path.join(os.getcwd(), 'test.csv'))

TRAIN_PATH = os.path.join(os.getcwd(), 'train_pp/')
TEST_PATH = os.path.join(os.getcwd(), 'test_pp/')

def read_img(img_path):
    img = misc.imread(img_path)
    return img


train_img = []
for img_path in tqdm(train['image_name'].values):
    train_img.append(read_img(TRAIN_PATH + img_path))
    
del img_path

x_train = np.array(train_img, np.float32) / 255.

mean_img = np.mean(x_train,axis=0)
std_img = np.std(x_train,axis=0)
x_train_norm = (x_train - mean_img) / std_img

# =============================================================================
# 
# #Use after load model
# np.save('mean_img.npy', mean_img) 
# np.save('std_img.npy', std_img)
# 
# =============================================================================

class_list = train['detected'].tolist()
Y_train = {k:v+1 for v,k in enumerate(set(class_list))}
y_train = [Y_train[k] for k in class_list]

y_train = to_categorical(y_train)


model = Sequential()
model.add(Convolution2D(64, (3,3), activation='relu', padding='same',input_shape = (128,128,1))) # if you resize the image above, change the shape
model.add(Convolution2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(16, (3,3), activation='relu', padding='same'))
model.add(Convolution2D(16, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Convolution2D(128, (3,3), activation='relu', padding='same'))
model.add(Convolution2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(y_train.shape[1], activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

x_train_norm = np.reshape(x_train_norm,(18577,128,128,1))
model.fit(x_train_norm, y_train, batch_size=100,epochs=10,validation_split=0.4)



test_img = []
for img in tqdm(test['image_name'].values):
    test_img.append(read_img(TEST_PATH + img))


x_test = np.array(test_img, np.float32) / 255.
x_test_norm = (x_test - mean_img) / std_img

predictions = model.predict(x_test_norm)
predictions = np.argmax(predictions, axis= 1)

y_maps = dict()
y_maps = {v:k for k, v in Y_train.items()}
pred_labels = [y_maps[k] for k in predictions]


sub = pd.DataFrame({'row_id':test.row_id, 'detected':pred_labels})
sub.to_csv('submission.csv', index=False)
