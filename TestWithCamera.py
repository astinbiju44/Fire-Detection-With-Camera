import numpy as np
import cv2
import pickle


width = 640
height = 480
threshold = 0.65 #minimum threshold for classify


cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

#load model
pickle_in = open("model_trained.p", "rb")
model = pickle.load(pickle_in)


