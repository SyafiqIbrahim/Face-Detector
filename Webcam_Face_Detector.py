import cv2

#Import pre-trained deepmachine learning data for face recognition
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Choose an image to detect face
webcam = cv2.VideoCapture(0)

#Iterate frames over and over
while True:
    #Read current frame
    successful_frame_read, frame = webcam.read()

    #Convert to greyscale
    greyscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detect face
    face_coordinates = trained_face_data.detectMultiScale(greyscaled_img)

    #Draw rectangles around face
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0),2)

    #Display image with faces detected
    cv2.imshow('Face Detection',frame)

    #Listen for key press for 1 milisecond, then move on
    key = cv2.waitKey(1)

    #Stop if Q key is pressed
    if key==81 or key==113:
        break

#Release the VideoCapture object
webcam.release()

print("Code completed")