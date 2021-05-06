import cv2
import urllib.request
import numpy as np
import imutils


url='http://192.168.1.4:8080/shot.jpg'

cam=cv2.VideoCapture(0)

def camera():
        _,img=cam.read()
        return img

#To use mobile camera download ipwebcam on your phone

def MobileCamera():
        imgpath=urllib.request.urlopen(url)    #this code help to open the url
        imgnp=np.array(bytearray(imgpath.read()),dtype=np.uint8)  #we are reading data from url in the from of bytearray
        img=cv2.imdecode(imgnp,-1)     #here imdecode helps to convert the array into image(pixels into images) #we are resizing the image to show
        img=imutils.resize(img,width=640,height=480)
        return img