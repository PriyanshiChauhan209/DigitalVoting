import cv2
import pickle 
import numpy as np 
import os 

if not os.path.exists('data/'):
    os.makedirs('data/')

video = cv2.VideoCapture(0)   #takes video from camera 
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') #haar_cascade is xml file that contains trained data 
faces_data=[] #array to store face data frames 
i=0
name= input ("Enter your aadhar number:")
framesTotal=51  
captureAfterFrame=2

while True:
    ret, frame = video.read()  # ret is boolean which is true if frame is successfully read and frame contains actual image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #converts image to grayscale
    faces= facedetect.detectMultiScale(gray, 1.3, 5) #datects images in gray scale and 'gray' is the grayscale image input 
    for (x,y,w,h) in faces :  #x, y → Top-left corner of the detected face.  w, h → Width and height of the detected face.


        crop_img = frame[y:y+h,x:x+w]  #This crops the face from the image using slicing.
        resized_img = cv2.resize(crop_img,(50,50))  #Resizes the cropped face to 50x50 pixels.
        if len (faces_data)<=framesTotal or i% captureAfterFrame==0:
            faces_data.append(resized_img)
        i=i+1
        #frame → Image to draw on.   str(len(faces_data)) → Text (number of stored face frames). (50, 50) → Position (x=50, y=50).  cv2.FONT_HERSHEY_COMPLEX, 1 → Font style and size.  (50, 50, 255), 1 → Text color (Blue-Red-Green in BGR format, thickness=1).
        cv2.putText(frame, str(len(faces_data)),(50,50),cv2.FONT_HERSHEY_COMPLEX, 1, (50,50,255),1)  
        # frame → Image on which to draw.
        # (x, y) → Top-left corner.
        # (x + w, y + h) → Bottom-right corner.
        # (50, 50, 255) → Color (Blue-Red-Green in BGR format).
        # 1 → Thickness of the rectangle.
        cv2.rectangle(frame, (x,y),(x+w, y+h),(50,50,255),1)


    cv2.imshow('frame',frame) #displays the captured image 
    k=cv2.waitKey(1) #cv2.waitKey(1) waits for 1 millisecond to check if a key is pressed.
    if k==ord('q') or len(faces_data)>=framesTotal:  #If the user presses 'q', the loop breaks.  If the collected face data reaches framesTotal (51), the loop stops.
        break

video.release()
cv2.destroyAllWindows()
# print(len(faces_data))
faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape((framesTotal, -1))
# print(faces_data)


if 'names.pkl' not in os.listdir('data/'):  #if the file does not already exists 
    names=[name]*framesTotal  #the name(Aadhar number) is repeated 51 times
    with open('data/names.pkl','wb')as f:
        pickle.dump(names,f) # save the new data in the file 
else:  #if the file already exists 
    with open('data/names.pkl','rb')as f:
        names=pickle.load(f) #convert in the numby list 
    names=names+[name]*framesTotal # append the data 
    with open('data/names.pkl','wb')as f:
        pickle.dump(names,f) # save in the file 

if 'faces_data.pkl' not in os.listdir('data/'):
    with open('data/faces_data.pkl','wb')as f:
        pickle.dump(faces_data,f)
else:
    with open('data/faces_data.pkl','rb')as f:
        faces=pickle.load(f)
    faces=np.append(faces,faces_data,axis=0)    
    with open('data/faces_data.pkl','wb')as f:
        pickle.dump(faces,f)