import numpy as np
from keras.models import load_model
import cv2

model = load_model('Chest_model.h5')
model.summary()

#Input Image
target_directory = 'Sample calss data/x.png'

#Load Image
def read_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img,(128,128))
    return img

test_img = read_img(target_directory)


#Pre-processing
mean_img = np.load('mean_img.npy')
std_img = np.load('std_img.npy')

x_test = np.array(test_img, np.float32) / 255.
x_test_norm = (x_test - mean_img) / std_img
x_test = x_test.reshape(1,128,128,3)


#Single Prediction
predictions = model.predict(x_test)
predictions = np.argmax(predictions, axis= 1)

#Label Dictionary
label_maps = {9:"Effusion", 4: "Fibrosis", 12: "Infiltration", 5: "Edema", 1: "Consolidation",
               13: "Emphysema", 8: "Atelectasis", 14: "Pleural_Thickening", 11: "Nodule",
                6: "Hernia", 3: "Mass", 10: "Pneumothorax", 2: "Pneumonia", 7: "Cardiomegaly"}

label = [label_maps[k] for k in predictions]

print(label)