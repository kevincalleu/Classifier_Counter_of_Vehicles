import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
#from tensorflow.keras import set_random_seed
#tf.random.set_seed(x)
import os
import cv2
from tqdm import tqdm

# np.random.seed(33)
# random.seed(33)
# set_random_seed(33)

img_rows, img_cols = 150, 150

TEST_DATADIR = r'test'
TRAIN_DATADIR = r'train'
class_names = ['liviano' , 'moto' , 'pesado']
num_classes = len(class_names)

train_data = []
test_data = []
x_train = []
y_train = []
x_test = []
y_test = []
x=[]
y=[]

# handle image format between PIL and OpenCV
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_cols, img_rows)
else:
    input_shape = (img_cols, img_rows, 3)


def create_train_data():
    for category in class_names:
        path = os.path.join(TRAIN_DATADIR,category)
        class_num = class_names.index(category)

        for img in tqdm(os.listdir(path)):  
            try:
                img_array = cv2.imread(os.path.join(path,img))  
                new_array = cv2.resize(img_array, (img_cols, img_rows))  
                train_data.append([new_array, class_num])  
            except Exception as e: 
                pass

    return train_data

def create_test_data():
    for category in class_names:
        path = os.path.join(TEST_DATADIR,category)
        class_num = class_names.index(category) 

        for img in tqdm(os.listdir(path)):  # iterate over each image
            try:
                img_array = cv2.imread(os.path.join(path,img))  # convert to array
                new_array = cv2.resize(img_array, (img_cols, img_rows))  # resize to normalize data size
                test_data.append([new_array, class_num])  # add this to our test_data
            except Exception as e:
                pass

    return test_data        

def get_array(data_array):
    x=[]
    y=[]
    for features,label in data_array:
        x.append(features)
        y.append(label)
    x = np.array(x).reshape(len(data_array), img_rows, img_cols, 3)
    y = np.array(y).reshape(len(data_array))

    return x, y

if __name__ == '__main__':
    with tf.device('/gpu:0'):
        train_data = create_train_data()
        x_train, y_train = get_array(train_data)
        test_data = create_test_data()
        x_test, y_test = get_array(test_data)

        # save data as numpy array
        np.save('x_train.npy', x_train)
        np.save('y_train.npy', y_train)
        np.save('x_test.npy', x_test)
        np.save('y_test.npy', y_test)
        
       