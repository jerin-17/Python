import cv2
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread('oscar.jpg')
imS = cv2.resize(img, (960, 640))
greyscale_img = cv2.cvtColor(imS,cv2.COLOR_BGR2GRAY)
face_coordinates = trained_face_data.detectMultiScale(greyscale_img)
#print(face_coordinates)
for face in face_coordinates:
    (x,y,w,h) = face
    cv2.rectangle(imS,(x,y),(x+w , y+h ),(0,255,0),2)
cv2.imshow('Actor',imS)
cv2.waitKey()
print('code completed')