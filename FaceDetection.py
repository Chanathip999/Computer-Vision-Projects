
import cv2
import os

# Korrekte Pfadbestimmung zur Haar-Cascade-Datei
cascPath = os.path.join(os.path.dirname(cv2.__file__), 'data', '/Users/chanathippaka/Desktop/VS CODE/haarcascade_frontalface_default.xml')
faceCascade = cv2.CascadeClassifier(cascPath)

if faceCascade.empty():
    print("Error loading cascade file. Check the path and file existence.")
    exit()

video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error: Could not open video device.")
    exit()

while True:
    ret, frame = video_capture.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
    cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

        
        