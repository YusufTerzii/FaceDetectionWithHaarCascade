import cv2
import numpy as np

vid = cv2.VideoCapture("C:/Users/yusuf/OpenCV/test_videos/faces.mp4")
face_cascade = cv2.CascadeClassifier("C:/Users/yusuf/OpenCV/haarCascade/frontalface.xml")

while True:
    ret,frame = vid.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.1,3)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow("faces",frame)
    if cv2.waitKey(5) & 0xFF == ord("q"):
        break

vid.release()
cv2.destroyAllWindows()