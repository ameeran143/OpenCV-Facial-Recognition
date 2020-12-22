import cv2
import numpy as np
import face_recognition
import os

# Facial recognition project

path = 'Pictures'
#creating list of all the images to be imported
images = []
classNames = []
myList = os.listdir(path) #all the images
print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}') #
    images.append(curImg) #storing all the images in a list
    classNames.append(os.path.splitext(cl)[0])  # gives "Bill Gates" instead of "Bill Gates.jpg"
print(classNames)

# Step 2: storing encodings: creating a function to find the encodings of all the images

def findEncodings(images):
    encodeList = [] # will store all the encodings

    #loop thorugh images
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # converting JPG to RGB
        encode = face_recognition.face_encodings(img)[0] #find the encodings
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)
print("Encoding complete")

# Step 3 - finding  matches between the encodings, the image to match it with is coming with the webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25) # small image, taking the webcam image and making it smaller to improve program effiency
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB) #converitng to RGB

    facesCurFrame = face_recognition.face_locations(imgS) # finding all the images in the small image
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)  # find the encoding of the webcam

#finding the matches
    for encodesFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodesFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodesFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis) # the match with the lowest distance is the correct one

        if matches[matchIndex]:
            name = classNames[matchIndex].upper() # this is the name
            print(name)
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1 *4,x2*4,y2*4,x1*4 # multiply by 4 to reverse the intial cut of the image by 4
            cv2.rectangle(img, (x1, y1),(x2, y2),(0,255,0), 2) #drawing a rectangle on the image
            cv2.rectangle(img,(x1,y2-35), (x2,y2), (0,255,0), cv2.FILLED)
            cv2.putText(img, name,(x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            # creating the filled rectangle to display the name of the person

    cv2.imshow('Webcam', img) # displaying webcam
    cv2.waitKey(1)