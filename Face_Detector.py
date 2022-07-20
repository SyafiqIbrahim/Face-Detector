#Import OpenCV
import cv2

#Import pre-trained deepmachine learning data for face recognition
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Choose an image to detect face
img = cv2.imread('Multiple People.webp')
#img = cv2.imread('One Person.webp')

#Convert to greyscale
greyscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Detect face
face_coordinates = trained_face_data.detectMultiScale(greyscaled_img)

#Draw rectangles around face
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0),2)

cv2.imshow('Face Detection',img)
cv2.waitKey()
