#!/usr/bin/env python
# coding: utf-8

# In[26]:

import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import csv
from keras.models import load_model
import numpy as np
from datetime import date
import random


# In[27]:


facedetect= cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')


# In[28]:


cap = cv2.VideoCapture(1)
cap.set(3,640)
cap.set(4,480)
font=cv2.FONT_HERSHEY_COMPLEX


# In[29]:


model = load_model('models/keras_model2.h5')


# In[30]:

#get date
today = date.today()

#split data
year = str(today.year)
month = str(today.month)
day = str(today.day)

isExist = os.path.exists(year)

if not isExist:
    os.mkdir(year)

    if not os.path.exists(f'{year}'+'/'+month):
        os.mkdir(year+'/'+month)

        if not os.path.exists(year+'/'+month+'/'+day):
            os.mkdir(year+'/'+month+'/'+day)

else:
    if not os.path.exists(f'{year}'+'/'+month):
        os.mkdir(year+'/'+month)

        if not os.path.exists(year+'/'+month+'/'+day):
            os.mkdir(year+'/'+month+'/'+day)
    else:
        if not os.path.exists(year+'/'+month+'/'+day):
            os.mkdir(year+'/'+month+'/'+day)

with open(year+'/'+month+'/'+day+'/'+str(today)+'.csv', mode='w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['ProbabilityValue','Name'])


    while True:
        success,imOrignal = cap.read()
        faces = facedetect.detectMultiScale(imOrignal,1.3,5)

        for x,y,w,h in faces:
            crop_img =imOrignal[y:y+h,x:x+h]
            img= cv2.resize(crop_img, (224,224))
            img = img.reshape(1,224,224,3)
            prediction = model.predict(img)

            a = [0,1,2,3,4,5,6]
        
            #classIndex= model.predict_classes(img)
            #probabilityValue = np.amax(prediction)
            #print(probabilityValue)
            #print(type(prediction))
            #print(prediction)
            probabilityValue = np.argmax(prediction)
            
            
            
            
            if  probabilityValue == 0:
                cv2.rectangle(imOrignal,(x,y),(x+w,y+h),(0,255,0), 2)
                cv2.rectangle(imOrignal,(x,y-40),(x+w, y),(0,255,0),-2)
                cv2.putText(imOrignal,'Angry',(x,y-10),font,0.75,(255,255,255),1,cv2.LINE_AA)
                #print("angry")

                writer.writerow([probabilityValue,'Angry'])

                cv2.putText(imOrignal,str(round(probabilityValue*100, 2))+"%" ,(x+150,y-10), font, 0.75, (255,0,0),2, cv2.LINE_AA)
                val = random.randint(1,4)
                os.system("afplay /music/"+val+".mp3")




            elif probabilityValue ==1:
                cv2.rectangle(imOrignal,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.rectangle(imOrignal, (x,y-40),(x+w, y), (0,255,0),-2)
                cv2.putText(imOrignal,'Disgust',(x,y-10), font, 0.75, (255,255,255),1, cv2.LINE_AA)
                #print("Disgust")
                #---------
                writer.writerow([probabilityValue,'Disgust'])
                val = random.randint(1,4)
                os.system("afplay /music/"+val+".mp3")

                cv2.putText(imOrignal,str(round(probabilityValue*100, 2))+"%" ,(x+150,y-10), font, 0.75, (255,0,0),2, cv2.LINE_AA)
            
            elif probabilityValue==2:
                cv2.rectangle(imOrignal,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.rectangle(imOrignal, (x,y-40),(x+w, y), (0,255,0),-2)
                cv2.putText(imOrignal,'fear',(x,y-10), font, 0.75, (255,255,255),1, cv2.LINE_AA)
                #print("fear") --------------
                
                writer.writerow([probabilityValue,'Fear'])
                val = random.randint(1,4)
                os.system("afplay /music/"+val+".mp3")

                cv2.putText(imOrignal,str(round(probabilityValue*100, 2))+"%" ,(x+150,y-10), font, 0.75, (255,0,0),2, cv2.LINE_AA)
                
            elif probabilityValue ==3:
                cv2.rectangle(imOrignal,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.rectangle(imOrignal, (x,y-40),(x+w, y), (0,255,0),-2)
                cv2.putText(imOrignal,'happy',(x,y-10), font, 0.75, (255,255,255),1, cv2.LINE_AA)
                #print("happy")

                writer.writerow([probabilityValue,'Happy'])

                cv2.putText(imOrignal,str(round(probabilityValue*100, 2))+"%" ,(x+150,y-10), font, 0.75, (255,0,0),2, cv2.LINE_AA)
            
            elif probabilityValue ==4:
                cv2.rectangle(imOrignal,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.rectangle(imOrignal, (x,y-40),(x+w, y), (0,255,0),-2)
                cv2.putText(imOrignal,'neutral',(x,y-10), font, 0.75, (255,255,255),1, cv2.LINE_AA)
                #print("neutral")

                writer.writerow([probabilityValue,'Neutral'])
                
                cv2.putText(imOrignal,str(round(probabilityValue*100, 2))+"%" ,(x+150,y-10), font, 0.75, (255,0,0),2, cv2.LINE_AA)
            
            elif probabilityValue==5:
                cv2.rectangle(imOrignal,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.rectangle(imOrignal, (x,y-40),(x+w, y), (0,255,0),-2)
                cv2.putText(imOrignal,'sad',(x,y-10), font, 0.75, (255,255,255),1, cv2.LINE_AA)
                #print("sad")
                
                writer.writerow([probabilityValue,'Sad'])

                cv2.putText(imOrignal,str(round(probabilityValue*100, 2))+"%" ,(x+150,y-10), font, 0.75, (255,0,0),2, cv2.LINE_AA)
                val = random.randint(1,4)
                os.system("afplay /music/"+val+".mp3")
            
            elif probabilityValue==6:
                cv2.rectangle(imOrignal,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.rectangle(imOrignal, (x,y-40),(x+w, y), (0,255,0),-2)
                cv2.putText(imOrignal,'suprise',(x,y-10), font, 0.75, (255,255,255),1, cv2.LINE_AA)
                #print("suprise")
                
                writer.writerow([probabilityValue,'Suprise'])

                cv2.putText(imOrignal,str(round(probabilityValue*100, 2))+"%" ,(x+150,y-10), font, 0.75, (255,0,0),2, cv2.LINE_AA)
    
        
        
        cv2.imshow("Result",imOrignal)
        k=cv2.waitKey(1)
        if k==ord('q'):
            break


# In[31]:


cap.release()
cv2.destroyAllWindows()


# In[ ]:




#https://pynative.com/python-get-month-name-from-number/
#https://www.programiz.com/python-programming/datetime/current-datetime
#https://stackoverflow.com/questions/12517451/automatically-creating-directories-with-file-output
#https://www.geeksforgeeks.org/create-a-directory-in-python/
#https://www.tutorialspoint.com/How-can-I-create-a-directory-if-it-does-not-exist-using-Python