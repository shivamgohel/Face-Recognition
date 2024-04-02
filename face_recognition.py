import numpy as np
import cv2
import os

########## KNN CODE ########
def distance(v1,v2):
    #Eucledian
    return np.sqrt(((v1-v2)**2).sum())

def knn(train,test,k=5):
    dist = []

    for i in range(train.shape[0]):
        #Get the vector and label
        ix = train[i, :-1]
        iy = train[i, -1]
        #compute the distance from the test point
        d = distance(test,ix)
        dist.append([d,iy])
    #Sort based on distance from the test point
    dk = sorted(dist,key=lambda x: x[0])[:k]
    #Retrieve only the labels
    labels = np.array(dk)[:, -1]

    #Get Frequencies of each label
    ouput = np.unique(labels,return_counts=True)
    #Find Max Frequency and corresponding label
    index = np.argmax(ouput[1])
    return ouput[0][index]
################################



#init camera
cap = cv2.VideoCapture(0)

#face detection
cascade_path = 'haarcascade_frontalface/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

skip = 0
dataset_path = './face_data/'
face_data = []
labels = []

class_id = 0   # labels for the given file
names = {}     # mapping between id - name


#Data Preparation
for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        #Create a mapping between class_id and name
        names[class_id] = fx[:-4]
        print("loaded "+fx)
        data_item = np.load(dataset_path+fx)
        face_data.append(data_item)

        #Create labels for class
        target = class_id*np.ones((data_item.shape[0],))
        class_id += 1
        labels.append(target)

face_dataset = np.concatenate(face_data,axis=0)
face_labels = np.concatenate(labels,axis=0).reshape((-1,1))

print(face_dataset.shape)
print(face_labels.shape)

trainset = np.concatenate((face_dataset,face_labels),axis=1)
print(trainset.shape)

#Testing

while True:
    ret,frame = cap.read()
    if ret == False:
        continue

    faces = face_cascade.detectMultiScale(frame,1.3,5)

    for face in faces:
        x,y,w,h = face

        #Get the face region of interest
        offset = 10
        face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))

        #Predicted Label (out)
        out = knn(trainset,face_section.flatten())

        #Display on the screen the name and recatangle around it
        pred_name = names[int(out)]
        cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

    cv2.imshow("Faces",frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



















