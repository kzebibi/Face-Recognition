import sqlite3

import cv2

# Detect the faces in camera
faceDetect = cv2.CascadeClassifier('../haarcascade/haarcascade_frontalface_default.xml')
camera = cv2.VideoCapture(0)  # 0 is for web camera


def insertorupdate(Id, Name, Age):
    conn = sqlite3.connect('../db/sqlite.db')
    cmd = 'SELECT * FROM Faces WHERE Id=' + str(Id)
    cursor = conn.execute(cmd);
    isRecordExists = 0
    for row in cursor:
        isRecordExists = 1
    if isRecordExists == 1:
        conn.execute('UPDATE Faces SET Name= ? WHERE Id= ?', (Name, Id,))
        conn.execute('UPDATE Faces SET Age= ? WHERE Id= ?', (Age, Id,))
    else:
        conn.execute('INSERT INTO Faces (Id, Name, Age) VALUES (?, ?, ?)', (Id, Name, Age,))

    conn.commit()
    conn.close()


# insert user defined values into table
Id = input('Enter ID: ')
Name = input('Enter Name: ')
Age = input('Enter Age: ')

insertorupdate(Id, Name, Age)

# Detect face in web camera coding

sampleNum = 0;
while True:
    ret, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        sampleNum = sampleNum + 1
        cv2.imwrite('../dataset/user.' + str(Id) + '.' + str(sampleNum) + '.jpg', gray[y:y + h, x:x + w])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.waitKey(100)
    cv2.imshow('Video', frame)
    cv2.waitKey(1)
    if sampleNum > 20:
        break

camera.release()
cv2.destroyAllWindows()
