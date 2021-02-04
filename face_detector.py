import cv2
from random import randrange

trained_face_data = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')

#trained_face_eye = cv2.CascadeClassifier('/data/haarcascade_eye.xml')

#img = cv2.imread('male.jpg')
#img = cv2.imread('female.jpg')
webcam = cv2.VideoCapture('video.mp4')


while True:
    successful_frame_read, frame = webcam.read()
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (randrange(256),randrange(256),randrange(256)),2)
    cv2.imshow('Video Face Detector', frame)
    key = cv2.waitKey(1)
    if key==81 or key==113: #ASCII value for q or Q
        break
webcam.release()


#grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

#for (x,y,w,h) in face_coordinates:
#    cv2.rectangle(img, (x,y), (x+w, y+h), (randrange(256),randrange(256),randrange(256)),2)

#(x,y,w,h) = face_coordinates[0]
#cv2.rectangle(img, (x, y),(x+w, y+h), (0,255,0),2)


# print(face_coordinates)

#cv2.imshow('AG', grayscaled_img)

#cv2.imshow('Face Detector', img)
#cv2.waitKey()

print("Code Completed") 


 