import cv2
import numpy
import numpy as np

#init camera
cap = cv2.VideoCapture(0)

#face detection
cascade_path = 'haarcascade_frontalface/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

skip = 0
face_data = []
dataset_path = './face_data/'

file_name = input("Enter the Name:")

while True:
    ret,frame = cap.read()

    if ret == False:
        continue

    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGRA2GRAY)

    faces = face_cascade.detectMultiScale(frame,1.3,5)
    #Pick the last face (because it is the largest face acc to area (f[2]*f[3]))
    faces = sorted(faces,key=lambda f:f[2]*f[3],reverse=True)

    for face in faces:
        x,y,w,h = face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

        #Extract (crop the required face) : Region of Interest
        offset = 10
        face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))

        skip += 1
        if skip%10==0:
            face_data.append(face_section)
            print(len(face_data))

    cv2.imshow("Frame",frame)
    #cv2.imshow("Face Section",face_section)

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

#Convert Our Face List Array Into a Numpy Array
face_data = np.array(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

#Save this Data Into File System

np.save(dataset_path+file_name+'.npy',face_data)
print("Data Successfully Saved at "+dataset_path+file_name+'.npy')

cap.release()
cv2.destroyAllWindows()



