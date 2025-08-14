# Face Recognition System Using KNN

This project implements a simple **face recognition system** using OpenCV for face detection and a custom K-Nearest Neighbors (KNN) algorithm for classification.

It consists of two main scripts:
- `face_data_collection.py`: Captures and saves face images of different individuals.
- `face_recognition.py`: Recognizes faces in real-time using the saved face data.

---

## Features

- Real-time face detection using Haar cascades  
- Collects face data and saves as `.npy` files  
- Implements KNN algorithm from scratch for face classification  
- Real-time face recognition with labeled bounding boxes  

---

## How It Works

### 1. Face Data Collection (`face_data_collection.py`)
- Uses webcam to capture video frames.  
- Detects faces using Haar cascade classifier.  
- Crops and resizes detected faces to a fixed size.  
- Saves collected face data as a NumPy array (`.npy`) labeled by person name.

### 2. Face Recognition (`face_recognition.py`)
- Loads saved face data and corresponding labels.  
- Prepares training data by concatenating features and labels.  
- Uses custom KNN to classify detected faces from webcam feed.  
- Displays recognized person's name and bounding box on the video.

---

## Dependencies

This project requires the following Python libraries:

- Python 3.x  
- OpenCV (`opencv-python`)  
- NumPy  

Install dependencies using pip:

```bash
pip install numpy opencv-python
