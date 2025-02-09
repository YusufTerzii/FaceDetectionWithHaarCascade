import cv2
import numpy as np

img=cv2.imread("C:/Users/yusuf/OpenCV/test_images/face.png")
face_cascade = cv2.CascadeClassifier("C:/Users/yusuf/OpenCV/haarCascade/frontalface.xml")

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray,1.3,7)

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()