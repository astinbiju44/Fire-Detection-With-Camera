import numpy as np
import cv2
import os
import time
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



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






#plotting graph to know how many images are in each class
numofsamples=[]
for x in range(0,noofclasses):
    print(len(np.where(y_train==x)[0]))  #y contains the class ids so it will give the number of images
    numofsamples.append(len(np.where(y_train==x)[0]))
print(numofsamples)
plt.figure(figsize=(10,5))
plt.bar(range(0,noofclasses),numofsamples)
plt.title("Number of images for each classes")
plt.xlabel("Class ID")
plt.ylabel("number of Images")
plt.show()






#data preprocessing
def preprocessing(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.equalizeHist(img) #this will help to equalize the distribution of lighting in each images
    img=img/255  #changing 0 to 255 into 0 to 1
    return img
x_train=np.array(list(map(preprocessing,x_train)))#map will take each image from input and send it to preprocessing function
                                                 #The result from funtion is stored in list using list.
                                                 #that list is converted to array
                                                 #this same as done in line 31
x_test=np.array(list(map(preprocessing,x_test)))
x_validation=np.array(list(map(preprocessing,x_validation)))






