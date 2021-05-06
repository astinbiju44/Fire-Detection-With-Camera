import numpy as np
import cv2
from keras.models import load_model
from BeepSound import beepsound
from CAMERA import camera,MobileCamera

width = 640
height = 480
threshold = 0.65 #minimum threshold for classify


cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

#model file loaded
model = load_model('fire_detection_model.h5')
print("Model Loaded Successfully")


#preprocess the camera image
def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img


count=0

while True:
    imgOriginal=MobileCamera()   # or  imgOriginal=MobileCamera()
    img = np.asarray(imgOriginal)
    img = cv2.resize(img, (32, 32))
    img = preProcessing(img)
    #cv2.imshow("Processsed Image", img)
    img = img.reshape(1, 32, 32, 1)
    #predicting
    classIndex = int(model.predict_classes(img))
    # print(classIndex)
    predictions = model.predict(img)
    # print(predictions)
    probVal = np.amax(predictions)
    #print(classIndex, probVal)
    if probVal > threshold:
        cv2.putText(imgOriginal, str(classIndex) + "   " + str(probVal),
                    (50, 50), cv2.FONT_HERSHEY_COMPLEX,
                    1, (0, 0, 255), 1)
        if classIndex==0:
            count+=1
        else:
            count=0
        if count==20:
            beepsound()

    print(count)
    cv2.imshow("Original Image", imgOriginal)
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break