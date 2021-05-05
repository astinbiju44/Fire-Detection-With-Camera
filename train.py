import numpy as np
import cv2
import os
import time
from sklearn.model_selection import train_test_split



#importing images
path="dataset"
mylist=os.listdir(path)
noofclasses=len(mylist)
print("Total number of classes detected : "+str(noofclasses))
images=[]
classno=[]
print("Importing classes.....")
time.sleep(2)
for x in range(0,noofclasses):
    mypiclist=os.listdir(path+"/"+str(x))
    for y in mypiclist:
        curimg=cv2.imread(path+"/"+str(x)+"/"+y)
        curimg=cv2.resize(curimg,(32,32))
        images.append(curimg)
        classno.append(x)
    print(x,end=",")
print("\n")



#converting into array
images=np.array(images)
classno=np.array(classno)
print(images.shape)
print(classno.shape)




#spliting the data
testratio=0.2
validationratio=0.2
x_train,x_test,y_train,y_test=train_test_split(images,classno,test_size=testratio)
x_train,x_validation,y_train,y_validation=train_test_split(x_train,y_train,test_size=validationratio)






