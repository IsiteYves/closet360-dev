import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load the PNG image
superhero_img = cv2.imread("test.png", cv2.IMREAD_UNCHANGED)

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Overlay superhero face onto detected faces
    for (x, y, w, h) in faces:
        roi = frame[y:y+h, x:x+w]
        superhero_face = cv2.resize(
            superhero_img, (w, h), interpolation=cv2.INTER_CUBIC)
        superhero_face = cv2.cvtColor(
            superhero_face, cv2.COLOR_BGRA2BGR)  # remove alpha channel
        frame[y:y+h, x:x+w] = cv2.addWeighted(superhero_face, 0.5, roi, 0.5, 0)

    # Display the resulting frame
    cv2.imshow("Live Face Detection", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam
cap.release()
cv2.destroyAllWindows()
