# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 12:40:45 2022

@author: kevin
"""
import cv2
import numpy as np

detected_classNames = []
input_size = 150
net = cv2.dnn.readNetFromTensorflow('frozen_model/simple_frozen_graph4.pb')
required_class_index = [0, 1, 2]
confThreshold =0.1

classNames = class_names = ['liviano' , 'moto' , 'pesado']
colors = np.random.randint(0, 255, size=(len(classNames), 3), dtype='uint8')

def postProcess(outputs,img):
    global detected_classNames 
    height, width = img.shape[:2]
    boxes = []
    classIds = []
    confidence_scores = []
    for output in outputs:
        for det in output:
            #print('out2 ',output)
            scores = det[:5]
            #print(scores)
            classId = np.argmax(scores)
            #print('ClassID ', classId)
            confidence = scores[classId]
            #print('Confidence ',confidence)
            
            if classId in required_class_index:
                if confidence > confThreshold:
                    #print(classId)
                    w, h = int(width), int(height)
                    x, y = int(width), int(height)
                    boxes.append([x, y, w, h])
                    classIds.append(classId)
                    confidence_scores.append(float(confidence))
    return classIds
        

def predict(image):
    img = cv2.imread(image)
    blob = cv2.dnn.blobFromImage(img, 1, (input_size, input_size), [0, 0, 0], 1, crop=False)
    
    # Set the input of the network
    net.setInput(blob)
    layersNames = net.getLayerNames()
    outputNames = [(layersNames[i-1]) for i in net.getUnconnectedOutLayers()]
    # Feed data to the network
    outputs = net.forward(outputNames)

    # Find the objects from the network output
    prediction = [postProcess(outputs,img)[0], classNames[postProcess(outputs,img)[0]]]
    print(prediction)
    return prediction

   
predict('capturas/captura112.jpg')