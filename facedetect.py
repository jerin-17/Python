import cv2
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
webcam = cv2.VideoCapture(0)

while True:
    successful_read,frame = webcam.read()
    greyscale_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face_coordinates = trained_face_data.detectMultiScale(greyscale_img)
    for face in face_coordinates:
        (x,y,w,h) = face
        cv2.rectangle(frame,(x,y),(x+w , y+h ),(0,255,0),2)
    cv2.imshow('Face Detection',frame)
    key =  cv2.waitKey(1)
    if key == 81 or key == 113:#ASCII for q|Q
        break
print("Face detection successful!")
