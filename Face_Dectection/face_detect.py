import cv2 as cv

# img = cv.imread('Resources/Photos/group 1.jpg')
# cv.imshow('Group of 5 people', img)


# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray People', gray)


# haar_cascade = cv.CascadeClassifier('F:\GITHUB\VisionCraft\Face_Dectection\haar_face.xml')

# faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)

# print(f'Number of faces found = {len(faces_rect)}')

# for (x,y,w,h) in faces_rect:
#     cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

# cv.imshow('Detected Faces', img)



# cv.waitKey(0)


cap = cv.VideoCapture(0)

while True:
    isTrue,frame=cap.read()

    if not isTrue:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    haar_cascade = cv.CascadeClassifier('F:\GITHUB\VisionCraft\Face_Dectection\haar_face.xml')

    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)

    print(f'Number of faces found = {len(faces_rect)}')

    for (x,y,w,h) in faces_rect:
        cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), thickness=2)

    cv.imshow('Detected Faces', frame)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break  

cap.release()
cv.destroyAllWindows()