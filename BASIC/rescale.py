import cv2 as cv
img=cv.imread('Resources/Photos/cat_large.jpg')

def rescaleFrame(frame , scale=0.75):
    