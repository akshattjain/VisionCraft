import cv2 as cv
# img=cv.imread('Resources/Photos/cat_large.jpg')

def rescaleFrame(frame, scale=0.75):
    # Images, Videos and Live Video
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width,height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

# Reading videos
capture = cv.VideoCapture('Resources/Videos/dog.mp4')

def changeRes(width,height):
    # Live video
    capture.set(3,width)
    capture.set(4,height)



while True:
    isTrue, frame = capture.read()

    frame_resized = rescaleFrame(frame, scale=.2)
    
    cv.imshow('Video', frame)
    cv.imshow('Video Resized', frame_resized)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()
    