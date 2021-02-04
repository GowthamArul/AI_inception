import cv2
from random import randrange

img_file = 'cars.jpg'
#video = cv2.VideoCapture('car.avi')
video = cv2.VideoCapture('pedestrians.avi')

car_tracker_file = './data/cars.xml'
car_tracker = cv2.CascadeClassifier(car_tracker_file)

pedestrain_tracker_file = './data/haarcascade_fullbody.xml'
pedestrain_tracker = cv2.CascadeClassifier(pedestrain_tracker_file)
 

while True:
    (read_successful, frame) = video.read()
    if read_successful:
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    cars = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrains = pedestrain_tracker.detectMultiScale(grayscaled_frame)

    for (x,y,w,h) in cars: 
        cv2.rectangle(frame, (x+1, y+2), (x+w, y+h), (randrange(256),randrange(256),randrange(256)), 2)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (randrange(256),randrange(256),randrange(256)), 2)

    for (x,y,w,h) in pedestrains: 
        cv2.rectangle(frame, (x,y), (x+w, y+h), (randrange(256),randrange(256),randrange(256)), 2)

    cv2.imshow("Car and Pedestrian", frame)
    key = cv2.waitKey(1)
     
    if key==81 or key==113: #ASCII value for q or Q
        break

video.release()

""" 
img = cv2.imread(img_file)

black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

car_tracker = cv2.CascadeClassifier(car_tracker_file)

cars = car_tracker.detectMultiScale(black_n_white)

for (x,y,w,h) in cars:
    cv2.rectangle(img, (x,y), (x+w, y+h), (randrange(256),randrange(256),randrange(256)), 2)

cv2.imshow("Car and Pedestrian", img)
"""

#cv2.waitKey()