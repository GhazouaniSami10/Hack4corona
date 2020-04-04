import cv2 as cv
import numpy
import tensorflow as tf
import cv2
from tensorflow.keras.optimizers import Adam
import numpy as np
import argparse
import os
from tensorflow import keras
from tensorflow.keras import layers


classNames = {0: 'normal', 1: 'person'}


class Detector:
    def __init__(self):
        global cvNet
        cvNet = cv.dnn.readNetFromTensorflow('model/tf_model.pb')
        print("Loaded Model from disk")
        
        
        
        
        
        # Reload the model from the 2 files we saved
        #with open('model/model.json') as json_file:
            #json_config = json_file.read()
        #new_model = keras.models.model_from_json(json_config)
        #new_model.load_weights('model/model2.h5')
        #print("Loaded Model from disk")
        # compile and evaluate loaded model
        #new_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    
    
    
    
    def detectObject(self, imName):
        img = cv.cvtColor(numpy.array(imName), cv.COLOR_BGR2RGB)
        cvNet.setInput(cv.dnn.blobFromImage(img, 0.007843, (300, 300), (127.5, 127.5, 127.5), swapRB=True, crop=False))
        #imgg = cv2.resize(img,(224,224))
        #imgg = np.reshape(img,[1,224,224,3])
        
        detections = cvNet.forward()
        
        #cols = img.shape[1]
        #rows = img.shape[0]


        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                class_id = int(detections[0, 0, i, 1])

                #xLeftBottom = int(detections[0, 0, i, 3] * cols)
                #yLeftBottom = int(detections[0, 0, i, 4] * rows)
                #xRightTop = int(detections[0, 0, i, 5] * cols)
                #yRightTop = int(detections[0, 0, i, 6] * rows)

                #cv.rectangle(img, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                             #(0, 0, 255))
                
                label = classNames[class_id] 
                
                
                
                #labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                #yLeftBottom = max(yLeftBottom, labelSize[1])
                
                
                font=cv2.FONT_HERSHEY_SIMPLEX
                cv.putText(img, label, (400,30),font,.7,(255,255,0),3,cv.LINE_AA)
                
                
                #cv2.putText(img2,str(txt_show),(400,30),font,.7,(255,255,0),3,cv2.LINE_AA)
   
                    #path = r'D:\corona\detect-covid19\00003359_000.png'
                    #img = cv2.imread(path).astype('float32')
                    #imgg = cv2.cvtColor(imgg, cv2.COLOR_BGR2RGB)
                    #imgg = cv2.resize(imgg,(224,224))
                    #imgg = np.reshape(imgg,[1,224,224,3])
                    #labels_str = ['covid', 'normal']
                    #predIdxs = new_model.predict(imgg, batch_size=1)
                    #predIdxs = np.argmax(predIdxs, axis=1)
                    #txt_show=labels_str[predIdxs[0]]
                    

        
        img = cv.imencode('.jpg', img)[1].tobytes()
        return img
