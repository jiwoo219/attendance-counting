import face_recognition
import argparse
import cv2
import os
import numpy as np
from edgetpu.detection.engine import DetectionEngine

# Construct the argument parser and parse the arguments.
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--dataset', required=True, help='path to input directory of faces + images')
ap.add_argument('-e', '--encodings', required=True, help='name of serialized output file of facial encodings')
args = vars(ap.parse_args())

# Init tpu engine.
DET_MODEL_PATH = './ssd_mobilnet_v2_face_quant_postprocess_edgetpu.tflite'
face_engine = DetectionEngine(DET_MODEL_PATH)

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Initialize the list of known face encoding and names.
known_face_encodings = []
known_face_names = []

# Load sample pictures and learn how to recognize it.
dirname = 'knowns'
files = os.listdir(dirname)
for filename in files:
    name, ext = os.path.splitext(filename)
    if ext == '.jpg':
        known_face_names.append(name)
        pathname = os.path.join(dirname, filename)
        img = face_recognition.load_image_file(pathname)
        face_encoding = face_recognition.face_encodings(img)[0]
        known_face_encodings.append(face_encoding)

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

def tpu_face_det(frame):
    CONFIDENCE_THRES = 0.05
    
    detection = face_engine.DetectWithInputTensor(input_tensor=res.reshape(-1), threshold=CONFIDENCE_THRES, top_k=1)

    if not detection:
        print('***no face found!***')
        return None
        
    box = (detection[0].bounding_box.flatten().tolist()) * np.array([w, h, w, h])
    (face_left, face_top, face_right, face_bottom) = box.astype('int')

    return [(face_top, face_right, face_bottom, face_left)]

def cv2_encoder(image, boxes):
    (face_top, face_right, face_bottom, face_left) = boxes[0]
    face_roi = image[face_top:face_bottom, face_left:face_right :]

    faceBlob = cv2.dnn.blobFromImage(
        face_roi, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
    embedder.setInput(faceBlob)

    encoding = embedder.forward()[0]

    return encoding

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    '''
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find al lthe faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known faces
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            min_distance = min(face_distances)

            name = "Unknown"

            if min_distance < 0.6 :
                best_match_index = np.argmin(face_distances)
                name = known_face_names[best_match_index]
            
            face_names.append(name)

            #print(name)
    '''
    if process_this_frame:
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        min_distance = min(face_distances)

        name = "Unknown"

        if min_distance < 0.6 :
            best_match_index = np.argmin(face_distances)
            name = know_face_names[best_match_index]

    boxes = tpu_face_det(frame)

    encoding = cv2_encoder(frame, boxes)

    face_encodings.append(encoding)
    face_names.append(name)
    

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        
        # Scale back up face locations
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows