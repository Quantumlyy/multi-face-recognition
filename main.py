# import libraries
import cv2
import face_recognition
from os import path, listdir
from sklearn import svm
import time

# Get a reference to webcam 
video_capture = cv2.VideoCapture(0)

encodings = []
names = []

# Initialize variables
face_locations = []

faces_dir = path.abspath("../faces") + "/"
faces_train_dir = listdir(faces_dir)

for person in faces_train_dir:
    pix = listdir(faces_dir + person)

    # Loop through each training image for the current person
    for person_img in pix:
        # Get the face encodings for the face in each image file
        face = face_recognition.load_image_file(
            faces_dir + person + "/" + person_img)
        face_bounding_boxes = face_recognition.face_locations(face)

        # If training image contains exactly one face
        if len(face_bounding_boxes) == 1:
            face_enc = face_recognition.face_encodings(face)[0]
            # Add face encoding for current image
            # with corresponding label (name) to the training data
            encodings.append(face_enc)
            names.append(person)
        else:
            print(person + "/" + person_img + " can't be used for training")

clf = svm.SVC(gamma='scale')
clf.fit(encodings, names)

frame_rate = 2
prev = 0

while True:
    time_elapsed = time.time() - prev
    # Grab a single frame of video
    ret, frame = video_capture.read()
    if time_elapsed > 1. / frame_rate:
        prev = time.time()
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = frame[:, :, ::-1]

        # Find all the faces in the current frame of video
        face_locations = face_recognition.face_locations(rgb_frame)

        no = len(face_locations)

        for i in range(no):
            top, right, bottom, left = face_locations[i]
            test_image_enc = face_recognition.face_encodings(rgb_frame, face_locations)[i]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            name = clf.predict([test_image_enc])
            cv2.putText(frame, *name, (right, bottom), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
