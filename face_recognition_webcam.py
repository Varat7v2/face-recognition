import FaceRecognition as FR
import cv2
import glob
import numpy as np
import pickle
import os, sys
import time

from facenet import facenet, detect_face

from face_detection import TensoflowFaceDector
from myFACENET import FACENET_EMBEDDINGS

# FACE_RECOGNITION = 'FACENET'
FACE_RECOGNITION = 'DLIB'
ADD_PERSONS = False

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

def face_distance(face_encodings, face_to_compare):
    """
    Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
    for each comparison face. The distance tells you how similar the faces are.

    :param faces: List of face encodings to compare
    :param face_to_compare: A face encoding to compare against
    :return: A numpy ndarray with the distance for each face in the same order as the 'faces' array
    """
    if len(face_encodings) == 0:
        return np.empty((0))

    return np.linalg.norm(face_encodings - face_to_compare, ord=None, axis=1)


def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.9):
    """
    Compare a list of face encodings against a candidate encoding to see if they match.

    :param known_face_encodings: A list of known face encodings
    :param face_encoding_to_check: A single face encoding to compare against the list
    :param tolerance: How much distance between faces to consider it a match. Lower is more strict. 0.6 is typical best performance.
    :return: A list of True/False values indicating which known_face_encodings match the face encoding to check
    """
    return list(face_distance(known_face_encodings, face_encoding_to_check) <= tolerance)

def cosine_similarity(dict_encoding, current_encoding):
    return (np.dot(dict_encoding, current_encoding)/(np.linalg.norm(dict_encoding)*np.linalg.norm(current_encoding)))

def cosine_distance(dict_encoding, current_encoding):
    return (1 - np.dot(dict_encoding, current_encoding)/(np.linalg.norm(dict_encoding)*np.linalg.norm(current_encoding)))

def euclidean_distance(dict_encoding, current_encoding):
    return (np.sqrt(np.sum((dict_encoding - current_encoding) ** 2)))


def main():
    PATH_TO_CKPT_FACE = 'models/myssd_mobilenet_v2_face.pb'
    tDetector = TensoflowFaceDector(PATH_TO_CKPT_FACE)

    PATH_TO_CKPT_FACENET_128D = 'models/facenet-20170511-185253.pb'
    PATH_TO_CKPT_FACENET_512D_9905 = 'models/facenet-20180408-102900-CASIA-WebFace.pb'
    PATH_TO_CKPT_FACENET_512D_9967 = 'models/faenet-20180402-114759-VGGFace2.pb'
    myfacenet = FACENET_EMBEDDINGS(PATH_TO_CKPT_FACENET_512D_9967)

    if ADD_PERSONS:
        print('Adding new persons encoding to dictionary ...')
        os.system('python add_persons.py')

    video_capture = cv2.VideoCapture(0)
    frame_width = int(video_capture.get(3))
    frame_height = int(video_capture.get(4))

    if FACE_RECOGNITION == 'FACENET':
        dict_known = deserialize_dict('data/faces_database_128D.pkl')
    elif FACE_RECOGNITION == 'DLIB':
        dict_known = deserialize_dict('data/faces_database_128D.pkl')
    
    known_face_names, known_face_encodings = list(), list()
    for key, values in dict_known.items():
        known_face_names.append(key)
        known_face_encodings.append(np.squeeze(values, axis=0).tolist())
        
    num_of_identities = len(known_face_names)
    # print(num_of_identities)
    
    while True:
        t1 = time.time()
        # Grab a single frame of video
        ret, frame = video_capture.read()
        
        boxes, scores, classes, num_detections = tDetector.run(frame)
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        face_locations = list()
        faces_cropped = list()
        face_boxes = list()

        for score, box in zip(scores, boxes):
            if score > 0.7:
                # ymin, xmin, ymax, xmax = box
                left = int(box[1]*frame_width)
                top = int(box[0]*frame_height)
                right = int(box[3]*frame_width)
                bottom = int(box[2]*frame_height)

                face_locations.append((top, right, bottom, left))
                face_boxes.append([left, top, right, bottom])
                cropped = frame[top:bottom, left:right]
                # cv2.imwrite('test.jpg', cropped)
                cropped = cv2.resize(cropped, (160,160), interpolation=cv2.INTER_LINEAR)
                faces_cropped.append(facenet.prewhiten(cropped))

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255),int(round(frame_height/150)), 2)


        num_curr_faces = len(faces_cropped)
        eucliDist_matrix = np.zeros((num_curr_faces, num_of_identities))

        if num_curr_faces > 0:

            if FACE_RECOGNITION is 'DLIB':
                face_encodings = FR.face_encodings(frame, face_locations)

                #***********************************************************************************************
                ### DLIB
                # Loop through each face in this frame of video
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    # See if the face is a match for the known face(s)
                    matches = FR.compare_faces(known_face_encodings, face_encoding)

                    name = "Unknown"

                    # If a match was found in known_face_encodings, just use the first one.
                    if True in matches:
                        first_match_index = matches.index(True)
                        name = known_face_names[first_match_index]

                    # Or instead, use the known face with the smallest distance to the new face
                    face_distances = FR.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                # Draw a label with a name below the face
                cv2.putText(frame, name, (left+5, top-15), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 255), 1)
                #************************************************************************************************

            if FACE_RECOGNITION is 'FACENET':
                current_encodings = myfacenet.run(faces_cropped)

                for i, current_encoding, facebox in zip(np.arange(len(current_encodings)).tolist(), current_encodings, face_boxes):
                    dist_updated = list()
                    for j, dict_name, dict_encodings in zip(np.arange(num_of_identities), known_face_names, known_face_encodings):
                        dist = list()
                        for dict_encoding in dict_encodings:
                            # dist.append(euclidean_distance(dict_encoding, current_encoding))
                            dist.append(cosine_similarity(dict_encoding, current_encoding))
                        # eucliDist_matrix[i][j] = np.min(np.asarray(dist, dtype=np.float32))
                        
                        dist_updated.append(max(dist))

                    if max(dist_updated) > 0.7:
                        name = known_face_names[np.argmax(dist_updated)]
                    else:
                        name = 'Unknown'

                    # cv2.rectangle(frame, (facebox[0], facebox[1]), (facebox[2], facebox[3]), (0,255,255), 3)
                    cv2.putText(frame, name, (facebox[0]+5, facebox[1]-15), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
        
                # for index in np.argmin(eucliDist_matrix, axis=1): # Row: 1, Cols: 0
                #     # print(np.min(eucliDist_matrix, axis=1))
                #     if np.min(eucliDist_matrix, axis=1) < 0.8:
                #         print(known_face_names[index])
                #     else:
                #         print('Unknown')

        time_lapse = time.time() - t1
        print(time_lapse*1000, 'milliseconds.')

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