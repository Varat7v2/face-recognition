import FaceRecognition as FR
import cv2
import glob
import numpy as np
import pickle
import os, sys

from faceDetection_frozenGraph import TensoflowFaceDector

def serialize_dict(dict_, filename):
    mydict = dict()
    # Load the embeddings of known people or check if there are any new people need to add in the list
    for image in glob.glob('known_persons/*'):
        name = image.split('/')[-1].split('.')[0]
        image = FR.load_image_file(image)
        encoding = FR.face_encodings(image)[0]
        mydict[name] = encoding

    # Serialize dictionary in binary format
    with open(filename, 'wb') as handle:
        pickle.dump(dict_, handle, protocol=pickle.HIGHEST_PROTOCOL)

def deserialize_dict(filename):
    # Deserialize dictionary
    with open(filename, 'rb') as handle:
        return pickle.load(handle)

def main():
    PATH_TO_CKPT = 'model/myssd_mobilenet_v2_face.pb'
    tDetector = TensoflowFaceDector(PATH_TO_CKPT)

    video_capture = cv2.VideoCapture(0)
    frame_width = int(video_capture.get(3))
    frame_height = int(video_capture.get(4))

    dict_known = deserialize_dict('data/known_persons.pkl')
    known_face_names, known_face_encodings = list(), list()
    for key, value in dict_known.items():
        known_face_names.append(key)
        known_face_encodings.append(value)

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()
        
        boxes, scores, classes, num_detections = tDetector.run(frame)
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)

        for score, box in zip(scores, boxes):
            if score > 0.7:
                # ymin, xmin, ymax, xmax = box
                left = int(box[1]*frame_width)
                top = int(box[0]*frame_height)
                right = int(box[3]*frame_width)
                bottom = int(box[2]*frame_height)

                # cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), int(round(frame_height/150)), 8)

                face_locations = [(top, right, bottom, left)]
                face_encodings = FR.face_encodings(frame, face_locations)

                # Loop through each face in this frame of video
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    # See if the face is a match for the known face(s)
                    matches = FR.compare_faces(known_face_encodings, face_encoding)

                    name = "Unknown"

                    # If a match was found in known_face_encodings, just use the first one.
                    # if True in matches:
                    #     first_match_index = matches.index(True)
                    #     name = known_face_names[first_match_index]

                    # Or instead, use the known face with the smallest distance to the new face
                    face_distances = FR.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255),int(round(frame_height/150)), 2)

        # Draw a label with a name below the face
        cv2.putText(frame, name, (left+5, top-15), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()