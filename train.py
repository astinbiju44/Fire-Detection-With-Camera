import numpy as np
import cv2
import os
import time



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










