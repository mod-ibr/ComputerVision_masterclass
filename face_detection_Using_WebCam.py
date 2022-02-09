import cv2

faceDetector = cv2.CascadeClassifier('F:/S-study/ML/04 - CcomputerVision_MasterClass/haarcascade_frontalface_default.xml')
# if you have other devices for camera change 0 to 1 and so on as whech your devices number.
video_Capture = cv2.VideoCapture(0)

while True:
    #Capture frame-By-Frame     
    ret,frame = video_Capture.read()
    image_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    detections = faceDetector.detectMultiScale(image_gray)
    
    #Draw a rectangle around face
    for (x,y,w,h) in detections:
        print(w,h)
        cv2.rectangle(frame,(x, y),(w+x,h+y),(0,255.0),3)
    
    #Display The Resulting Frame
    cv2.imshow('Video', frame)
    if cv2.waitKey(1)& 0xFF==ord('q'):
        break
    # when Every Thing Is Done, Release the Capture
video_Capture.release()
cv2.destroyAllWindows()