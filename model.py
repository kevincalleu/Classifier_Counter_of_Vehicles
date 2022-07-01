from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.regularizers import l2

# with tf.device('/gpu:0'):
#     img_rows, img_cols = 227, 227  # image size
#     batch_size = 32
#     epochs = 11
#     num_classes = 3 
#     l2_reg = 0 # regularization
#     class_names = ['liviano', 'moto', 'pesado']
#     x_train = np.load('x_train.npy')
#     y_train = np.load('y_train.npy')
#     x_test = np.load('x_test.npy')
#     y_test = np.load('y_test.npy')

#     (x_train, x_test, y_train, y_test) = train_test_split(x_train, y_train, test_size=0.20)
    
#     model = keras.Sequential()
#     model.add(keras.layers.Conv2D(filters=96, kernel_size=(11, 11), 
#                             strides=(4, 4), activation="relu", 
#                             input_shape=(227, 227, 3)))
#     model.add(keras.layers.BatchNormalization())
#     model.add(keras.layers.MaxPool2D(pool_size=(3, 3), strides= (2, 2)))
#     model.add(keras.layers.Conv2D(filters=256, kernel_size=(5, 5), 
#                             strides=(1, 1), activation="relu", 
#                             padding="same"))
#     model.add(keras.layers.BatchNormalization())
#     model.add(keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
#     model.add(keras.layers.Conv2D(filters=384, kernel_size=(3, 3), 
#                             strides=(1, 1), activation="relu", 
#                             padding="same"))
#     model.add(keras.layers.BatchNormalization())
#     model.add(keras.layers.Conv2D(filters=384, kernel_size=(3, 3), 
#                             strides=(1, 1), activation="relu", 
#                             padding="same"))
#     model.add(keras.layers.BatchNormalization())
#     model.add(keras.layers.Conv2D(filters=256, kernel_size=(3, 3), 
#                             strides=(1, 1), activation="relu", 
#                             padding="same"))
#     model.add(keras.layers.BatchNormalization())
#     model.add(keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
#     model.add(keras.layers.Flatten())
#     model.add(keras.layers.Dense(4096, activation="relu"))
#     model.add(keras.layers.Dropout(0.5))
#     model.add(keras.layers.Dense(10, activation="softmax"))
#     model.compile(loss='sparse_categorical_crossentropy', 
#                   optimizer=tf.optimizers.SGD(lr=0.001), 
#                   metrics=['accuracy'])
#     model.summary()

#     model.compile(optimizer='adam', 
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

#     trainmodel = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test),batch_size=batch_size)

#     model.save('modelo.h5')
#     test_loss, test_acc = model.evaluate(x_test, y_test)
#     print('Test accuracy:', test_acc)
    
#     history = trainmodel.history
#     history.keys()

#     acc = history['accuracy']
#     val_acc = history['val_accuracy']
#     loss = history['loss']
#     val_loss = history['val_loss']

#     epochs = range(1, len(acc) + 1)

#     # "bo" is for "blue dot"
#     plt.plot(epochs, loss, 'bo', label='Training loss')
#     # b is for "solid blue line"
#     plt.plot(epochs, val_loss, 'b', label='Testing loss')
#     # "bo" is for "blue dot"
#     plt.plot(epochs, acc, 'go', label='Training accuracy')
#     # b is for "solid blue line"
#     plt.plot(epochs, val_acc, 'g', label='Testing accuracy')
#     plt.title('Graph of training and testing loss/accuracy vs number of epochs')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss/Accuracy')
#     plt.legend()

#     plt.savefig('train-test.png')
#     plt.show()

with tf.device('/gpu:0'):
    img_rows, img_cols = 150, 150  
    batch_size = 32
    epochs = 25
    num_classes = 3 
    l2_reg = 0 
    class_names = ['liviano', 'moto', 'pesado']
    x_train = np.load('x_train.npy')
    y_train = np.load('y_train.npy')
    x_test = np.load('x_test.npy')
    y_test = np.load('y_test.npy')

    (x_train, x_test, y_train, y_test) = train_test_split(x_train, y_train, test_size=0.15)

    #define model
    model = keras.Sequential([
        keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=x_train.shape[1:], kernel_regularizer=l2(l2_reg)),
        # keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.AveragePooling2D(pool_size=2, strides=2),
        keras.layers.Dropout(0.2),
    
    # keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
    # keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
    # keras.layers.BatchNormalization(),
    # keras.layers.AveragePooling2D(pool_size=2),
    # keras.layers.Dropout(0.5),   
    
    # keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
    # keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
    # keras.layers.BatchNormalization(),
    # keras.layers.AveragePooling2D(pool_size=2),
    # keras.layers.Dropout(0.5),   
    
    # keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),
    # keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),
    # keras.layers.AveragePooling2D(pool_size=2),
    # keras.layers.BatchNormalization(),
    # keras.layers.Dropout(0.5),   
    
        keras.layers.Flatten(),
        keras.layers.Dense(32, activation=tf.nn.relu),
        keras.layers.Dropout(0.5),    
        keras.layers.BatchNormalization(),
        keras.layers.Dense(num_classes, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    trainmodel = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test),batch_size=batch_size)

    model.save('modelo.h5')
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('Test accuracy:', test_acc)
    
    history = trainmodel.history
    history.keys()

    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']

    epochs = range(1, len(acc) + 1)

    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Testing loss')
    # "bo" is for "blue dot"
    plt.plot(epochs, acc, 'go', label='Training accuracy')
    # b is for "solid blue line"
    plt.plot(epochs, val_acc, 'g', label='Testing accuracy')
    plt.title('Graph of training and testing loss/accuracy vs number of epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss/Accuracy')
    plt.legend()

    plt.savefig('train-test.png')
    plt.show()