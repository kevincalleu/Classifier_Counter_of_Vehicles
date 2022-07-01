# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 20:48:01 2022

@author: Kevin Calle
"""

import cv2
import numpy as np
import cnn
from time import sleep

def getvehiculos(video, posicion_l, x_inicio,x_fin, offset, long_marco):
    
    x_inicio = x_inicio
    x_fin = x_fin
    offset= offset 
    vehiculos = []
    limites = []
    carros= 0
    moto = 0
    liviano = 0
    pesado = 0
    
    # Longitud minima de altura y ancho de los marcos     
    ancho_min = long_marco 
    alto_min = long_marco 
    
    # FPS
    delay= 60 

    cap = cv2.VideoCapture(video)
        
    # Creacion del Substractor
    fgbg = cv2.createBackgroundSubtractorMOG2()
    #fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

    kernelOp = np.ones((3, 3), np.uint8)
    # kernelOp = np.ones((5, 5), np.uint8)
    kernelCl = np.ones((11, 11), np.uint8)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    count=0
    
    while True:
        ret , frame = cap.read()
        #cv2.imshow('Frame', frame)
        
        # pausa = float(1/delay)
        # sleep(pausa) 
        
        try:
            
            fgmask = fgbg.apply(frame)
            #cv2.imshow('Substraccion de Fondo', fgmask)
            ret, imBin = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
            #cv2.imshow('Imagen binaria', imBin)
            
            # Filtro de apertura
            mask = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernelOp)
            
            # Filtro de clausura
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernelCl)
            
            #cv2.imshow('Image Threshold', cv2.resize(fgmask, (400, 300)))
            #cv2.imshow('Masked Image', cv2.resize(mask, (400, 300)))
            
        except:
            break
        
        # Extraccion de contornos
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        frame2 = frame.copy()
        
        # Linea contadora
        cv2.line(frame, (x_inicio, posicion_l), (x_fin, posicion_l), (255,127,0), 3) 
        
        for(_,c) in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(c)
            valid_contours = (w >= ancho_min) and (h >= alto_min)
            if not valid_contours:
                continue

            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)        
            center = getCentroid(x, y, w, h)
            vehiculos.append(center)
            limites.append((x,y,x+w,y+h))
            cv2.circle(frame, center, 6, (0, 0,255), -1)

            for (x, y) in vehiculos:
                if y<(posicion_l+offset) and y>(posicion_l-offset) and x>=x_inicio and x<=x_fin:
                    carros+=1
                    i = vehiculos.index((x,y))
                    box= limites[i]
                    x1 = box[1]-10
                    if x1 <0:
                        x1 = 0
                    x2 = box[3] + 10
                    if x2 > frame2.shape[0]:
                         x2 = frame2.shape[0]
                    y1 = box[0] - 10
                    if y1 <0:
                        y1 = 0
                    y2 = box[2] + 10
                    if y2 > frame2.shape[1]:
                         y2 = frame2.shape[1]
                    
                    cv2.imwrite('capturas/' + "\\captura%d.jpg" % count, frame2[x1:x2, y1:y2])
                    
                    # Se aplica la red neuronal para predecir el tipo de vehiculo
                    prediction = cnn.predict('capturas/' + "\\captura%d.jpg" % count)[1]
                    
                    print(prediction)
                    if prediction == 'moto':
                        moto += 1
                    elif prediction == 'liviano':
                        liviano += 1
                    elif prediction == 'pesado':
                        pesado += 1
                        
                        
                    count = count + 1
                    cv2.line(frame, (x_inicio, posicion_l), (x_fin, posicion_l), (0,115,255), 3) 
                    vehiculos.remove((x,y))
                    limites.pop(i)
                    
        moto_c = 'Moto: ' + str(moto)
        liviano_c = 'Liviano: ' + str(liviano)
        pesado_c = 'Pesado: ' + str(pesado)        
        cv2.putText(frame, moto_c, (10, 40), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, liviano_c, (10, 90), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, pesado_c, (10, 140), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.imshow('Conteo y Clasificacion', cv2.resize(frame, (640, 480))) 
               
        k = cv2.waitKey(10) & 0xff
        if k == 27:
            break
        
    cap.release()
    cv2.destroyAllWindows()
    
def getCentroid(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx,cy                

if __name__ == '__main__':
    video = 'video.mp4'
    posicion_l = 380
    x_inicio = 0
    x_fin = 1280
    offset = 5
    long_marco = 30    
   
    getvehiculos(video, posicion_l, x_inicio, x_fin, offset, long_marco)()
    