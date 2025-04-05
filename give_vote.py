from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle 
import numpy as np 
import os 
import csv 
import time 
from datetime import datetime 
from win32com.client import Dispatch

def speak(str1):
    speak = Dispatch(("SAPI.SpVoice"))
    speak.Speak(str1)

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if not os.path.exists('data/'):
    os.makedirs('data/')

with open('data/names.pkl', 'rb')as f:
    LABELS = pickle.load(f)

with open('data/faces_data.pkl', 'rb')as f:
    FACES = pickle.load(f)

FACES = np.array([face.flatten() for face in FACES])
LABELS = np.array([label[0] if isinstance(label, list) else label for label in LABELS])
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

imgBackground = cv2.imread("bg_img.jpg")

# Move function definition outside loop
def check_if_exists(value):
    try:
        with open("Votes.csv", "r") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row and row[0] == value:
                    return True
    except FileNotFoundError:
        print("File not found or unable to open the CSV file")
    return False

COL_NAMES = ["Name", "Party", "Date", "Time"]

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resized_img)

        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")
        exist = os.path.isfile("Votes.csv")

        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
        cv2.putText(frame, str(output[0]), (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

        attendance = [output[0], timestamp]
        resized_frame = cv2.resize(frame, (220, 250))
        imgBackground[83:83+250, 19:19+220] = resized_frame

        # Moved this inside the loop, after output is defined
        voter_exist = check_if_exists(output[0])
        if voter_exist:
            speak("YOU HAVE ALREADY VOTED")
            vote_casted=True
            break

        k = cv2.waitKey(1)

        if k == ord('1'):
            speak("YOUR VOTE HAS BEEN RECORDED")
            time.sleep(3)
            with open("Votes.csv", "a", newline='') as csvfile:
                writer = csv.writer(csvfile)
                if not exist:
                    writer.writerow(COL_NAMES)
                writer.writerow([output[0], "BJP", date, timestamp])
            speak("THANK YOU FOR PARTICIPATING IN THE ELECTIONS")
            break

        if k == ord('2'):
            speak("YOUR VOTE HAS BEEN RECORDED")
            time.sleep(3)
            with open("Votes.csv", "a", newline='') as csvfile:
                writer = csv.writer(csvfile)
                if not exist:
                    writer.writerow(COL_NAMES)
                writer.writerow([output[0], "Congress", date, timestamp])
            speak("THANK YOU FOR PARTICIPATING IN THE ELECTIONS")
            break

        if k == ord('3'):
            speak("YOUR VOTE HAS BEEN RECORDED")
            time.sleep(3)
            with open("Votes.csv", "a", newline='') as csvfile:
                writer = csv.writer(csvfile)
                if not exist:
                    writer.writerow(COL_NAMES)
                writer.writerow([output[0], "AAP", date, timestamp])
            speak("THANK YOU FOR PARTICIPATING IN THE ELECTIONS")
            break

        if k == ord('4'):
            speak("YOUR VOTE HAS BEEN RECORDED")
            time.sleep(3)
            with open("Votes.csv", "a", newline='') as csvfile:
                writer = csv.writer(csvfile)
                if not exist:
                    writer.writerow(COL_NAMES)
                writer.writerow([output[0], "Rashtravadi Cong.", date, timestamp])
            speak("THANK YOU FOR PARTICIPATING IN THE ELECTIONS")
            break
    if vote_casted:
        break  # exit while loop

    cv2.imshow('frame', imgBackground)

video.release()
cv2.destroyAllWindows()
