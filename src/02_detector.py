import sqlite3

import cv2

facedetect = cv2.CascadeClassifier("../haarcascade/haarcascade_frontalface_default.xml")
camera = cv2.VideoCapture(0)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("../recognizer/trainingData.yml")


def getProfile(Id):
    conn = sqlite3.connect("../db/sqlite.db")
    cursor = conn.execute("SELECT * FROM Faces WHERE Id=?", (Id,))
    profile = None
    for row in cursor:
        profile = row
    conn.close()
    return profile


while True:
    ret, frame = camera.read();
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5);
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, conf = recognizer.predict(gray[y:y + h, x:x + w])
        profile = getProfile(id)
        print(profile)
        if (profile != None):
            cv2.putText(frame, "Name : " + str(profile[1]), (x, y + h + 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 127),
                        2)
            cv2.putText(frame, "Age : " + str(profile[2]), (x, y + h + 45), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 127),
                        2)

    cv2.imshow("Face", frame);
    if (cv2.waitKey(1) == ord('q')):
        break;

cam.release()
cv2.destroyAllWindows()
