import numpy as np
import pickle
import cv2
import os, glob, sys
import dlib

# import FaceRecognition as FR
from tqdm import tqdm
import shutil 
import pandas as pd
import random
from pathlib import Path
import matplotlib.pyplot as plt

from facenet import facenet, detect_face
from myFACENET import FACENET_EMBEDDINGS
from headpose.headpose import HeadPoseEstimation

from face_detection import TensoflowFaceDector

# PATH_TO_CKPT = 'models/myssd_mobilenet_v2_face.pb'
PATH_TO_CKPT = 'models/frozen_graph_face_512x512.pb'
FACE_RECOGNITION = 'facenet'
FACE_EMBEDDINGS = '128D' #128D or 512D
# src_path = 'data/lfw'
embedding_data = 'data/lfw_embeddings'
test_data = 'data/lfw_test'
TEST_SPLIT = 0.5
src_path = 'my_known_persons_test'
ADD_FROM_WEBCAM = False
# src_path = embedding_data
DLIB_LANDMARKS_MODEL = 'models/shape_predictor_68_face_landmarks.dat'
PATH_TO_CKPT_FACENET_128D = 'models/facenet-20170511-185253.pb'
PATH_TO_CKPT_FACENET_512D_9905 = 'models/facenet-20180408-102900-CASIA-WebFace.pb'
PATH_TO_CKPT_FACENET_512D_9967 = 'models/faenet-20180402-114759-VGGFace2.pb'

# OBJECT INITIALIZATION
tDetector = TensoflowFaceDector(PATH_TO_CKPT)
dlib_landmarks = dlib.shape_predictor(DLIB_LANDMARKS_MODEL)

if FACE_EMBEDDINGS == '128D':
    facenet_model = PATH_TO_CKPT_FACENET_128D
elif FACENET_EMBEDDINGS == '512D':
    facenet_model = PATH_TO_CKPT_FACENET_512D_9967

if FACE_RECOGNITION == 'facenet':
    myfacenet = FACENET_EMBEDDINGS(facenet_model)

headpose = HeadPoseEstimation()

def serialize_dict(filename, mydict):
    # Serialize dictionary in binary format
    with open(filename, 'wb') as handle:
        pickle.dump(mydict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def deserialize_dict(filename):
    # Deserialize dictionary
    with open(filename, 'rb') as handle:
        return pickle.load(handle)

def test_data_split():
    for dir_name in tqdm(os.listdir(src_path)):
        # print(len(list(file for file in glob.glob(os.path.join(src_path, dir_name)+'/*'))))
        if len(list(file for file in glob.glob(os.path.join(src_path, dir_name)+'/*'))) > 3:
            # print('Entered')
            if not os.path.exists(test_data+'/'+dir_name):
                os.makedirs(test_data+'/'+dir_name)
            if not os.path.exists(os.path.join(embedding_data, dir_name)):
                os.makedirs(os.path.join(embedding_data, dir_name))

            files = list(file.split('/')[-1] for file in glob.glob(os.path.join(src_path, dir_name) + '/*'))
            files_copy = files.copy()
            test_size = int(TEST_SPLIT * len(files))
            test_files = list()
            embedding_files = list()

            for i in range(test_size):
                file_selected = random.choice(files)
                test_files.append(file_selected)
                files.remove(file_selected)
            embedding_files = list(set(files_copy).difference(set(test_files)))

            for test_file in test_files:
                os.system('cp -r {}/{}/{} {}/{}'.format(src_path, dir_name, test_file, test_data, dir_name))
            for embedding_file in embedding_files:
                os.system('cp -r {}/{}/{} {}/{}'.format(src_path, dir_name, embedding_file, embedding_data, dir_name))

def find_landmarks(frame, boxes):
    # DLIB FACIAL LANDMARKS
    grayImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for box in boxes:
        # print(box)
        frame = cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0,0,255), 2)
        dlibRect = dlib.rectangle(box[0], box[1], box[2], box[3])
        facial_landmarks = dlib_landmarks(grayImg, dlibRect)

        for i in range(68):
            # print(facial_landmarks.part(i).x, facial_landmarks.part(i).y)
            cv2.circle(frame, (facial_landmarks.part(i).x, facial_landmarks.part(i).y), 2, (0,0,255), -1)

    return frame, facial_landmarks

def detect_face(frame, dir_name, count):
    frame_height, frame_width, _ = frame.shape

    boxes, scores, classes, num_detections = tDetector.run(frame)
    boxes = np.squeeze(boxes)
    scores = np.squeeze(scores)
    faces = list()
    myboxes = list()
    scores_updated = list(score for score in scores if score > 0.6)
    score_count = len(scores_updated)
    boxes_updated = list(boxes[i] for i in range(score_count))

    if score_count == 1:
        for score, box in zip(scores_updated, boxes_updated):
            left = int(box[1]*frame_width)
            top = int(box[0]*frame_height)
            right = int(box[3]*frame_width)
            bottom = int(box[2]*frame_height)

            cropped_img = frame[top:bottom, left:right]
            cropped_img = cv2.resize(cropped_img, (160,160), interpolation=cv2.INTER_LINEAR)
            cropped = facenet.prewhiten(cropped_img)
            myboxes = [left, top, right, bottom]
            # Saving cropped face
            # if not os.path.exists(os.path.join(dst_path, dir_name)):
            #     os.makedirs(os.path.join(dst_path, dir_name))
            # cv2.imwrite(os.path.join(dst_path, dir_name) + '/' + str(count) + '_test.jpg', cropped_img)

            # cv2.rectangle(frame, (left, top), (right, bottom),(0, 255, 0), int(round(frame_height/150)), 8)
            # TODO: For multi-faces we have to CLUSTERING

        return myboxes, cropped #TODO: when no faces detected send NONE; This is only for single face in a image
    else:
        return myboxes, None

def get_facial_embeddings(frame, myfacenet, faces, facial_landmarks):
    if FACE_RECOGNITION == 'facenet':
        return myfacenet.run(faces)
    if FACE_RECOGNITION == 'dlib':
        return FR.myface_encodings(frame, facial_landmarks)[0]

def cosine_similarity(dict_encoding, current_encoding):
    return (np.dot(dict_encoding, current_encoding)/(np.linalg.norm(dict_encoding)*np.linalg.norm(current_encoding)))

def cosine_distance(dict_encoding, current_encoding):
    return (1 - np.dot(dict_encoding, current_encoding)/(np.linalg.norm(dict_encoding)*np.linalg.norm(current_encoding)))

def eucledian_distance(dict_encoding, current_encoding):
    return np.sqrt(np.sum((dict_encoding - current_encoding) ** 2))

def define_pose_category(angle):
    if angle >= -30 and angle <= 30:
        return 'category_1'
    elif angle >= 30 and angle <= 60:
        return 'category_2'
    elif angle >= -60 and angle <= -30:
        return 'category_2'
    elif angle >= 60 and angle <= 90:
        return 'category_3'
    elif angle >= -90 and angle <= -60:
        return 'category_3'

def find_threshold_stats():
    mydict_list = list()
    for dir_name in tqdm(os.listdir(src_path)):
        if len(list(f for f in glob.glob(os.path.join(src_path, dir_name)+'/*'))) > 2:
            # print('Entered')
            cropped_faces = list()
            faces_embeddings = list()
            facialLandmarks = list()
            facialHeadposes = list()
            mydict = dict()
            count = 0
            for file in glob.glob(os.path.join(src_path, dir_name)+'/*'):
                filename = file.split('/')[-1]
                frame = cv2.imread(file)
                face_boxes, cropped = detect_face(frame, dir_name, count)
                if cropped is not None:
                    cropped_faces.append(cropped)

                    # FIND THE FACIAL LANDMARKS
                    frame_landmarks, dlibLandmarks = find_landmarks(frame, [face_boxes])
                    facialLandmarks.append(dlibLandmarks)
                    # cv2.imwrite('data/processed_images/'+filename, frame)

                    # HEADPOSE ESTIMATION
                    frame_headpose, angles = headpose.findHeadPose(frame, dlibLandmarks, face_boxes)
                    facialHeadposes.append(angles)
                    # print('PITCH: {}, YAW: {}, ROLL: {}'.format(angles[0], angles[1], angles[2]))
                    cv2.imwrite('data/processed_images/'+filename, frame_headpose)
                count += 1
            if len(cropped_faces) > 0:
                faces_embeddings.append(get_facial_embeddings(frame, myfacenet, cropped_faces, facialLandmarks))
                mydict['face_id'] = dir_name
                mydict['face_embeddings'] = np.asarray(faces_embeddings, dtype=np.float32)
                mydict['landmarks'] = np.asarray(facialLandmarks)
                mydict['headpose'] = np.asarray(facialHeadposes, dtype=np.float32)
                mydict_list.append(mydict)

    # Check if dictionary is empty
    if len(mydict_list) is 0:
        raise Exception("The size of the dictionary: mydict is 0")
    else:
        print('dictionary size: {}'.format(len(mydict_list)))
        # print(mydict_list[0])

    database_face_ids = list()
    database_face_embeddings = list()
    database_landmarks = list()
    database_headpose = list()
    database_headpose_yaw = list()
    for dict_ in mydict_list:
        database_face_ids.append(dict_['face_id'])
        database_face_embeddings.append(np.squeeze(dict_['face_embeddings'], axis=0))
        database_landmarks.append(dict_['landmarks'])
        database_headpose.append(dict_['headpose'])
    
    # Visualizing headpose data more elaboratively
    # headpose[0] --> Pitch, headpose[1] --> Yaw, headpose[2] --> Roll
    print('Extracting only YAW anlge from headpose array ...')
    for idx, pose_per_person in zip(database_face_ids, database_headpose):
        # print(idx)
        pose_person = list()
        for pose_per_face in pose_per_person:
            pose_person.append(pose_per_face[1])
            # print(pose_per_face[1])
        database_headpose_yaw.append(pose_person)
    # print(database_headpose_yaw)

    # stats_menu = ['min', 'max', 'mean', 'std', 'variance']
    threshold_auto_dict = dict()
    threshold_cross_dict = dict()
    threshold_dict = dict()
    threshold_auto_min = list()
    threshold_cross_min = list()
    threshold_auto_avg = list()
    threshold_cross_avg = list()

    for i, name, dict_encodings, myheadpose in tqdm(zip(np.arange(len(database_face_ids)), database_face_ids, database_face_embeddings, database_headpose_yaw)):
        # print(name, dict_encodings, myheadpose)
        # CATEGORY-1
        auto_dist_1 = list()
        cross_dist_1 = list()
        # CATEGORY-2
        auto_dist_2 = list()
        cross_dist_2 = list()
        # CATEGORY-3
        auto_dist_3 = list()
        cross_dist_3 = list()

        auto_min, cross_min = list(), list()
        auto_max, cross_max = list(), list()
        auto_avg, cross_avg = list(), list()
        auto_var, cross_var = list(), list()
        auto_std, cross_std = list(), list()
        for j, dict_encodings_copy, myheadpose_copy in zip(np.arange(len(database_face_ids)), database_face_embeddings, database_headpose_yaw):
            for k, (dict_encoding, pose_yaw)in enumerate(zip(dict_encodings, myheadpose)):
                for l, (dict_encoding_copy, pose_yaw_copy) in enumerate(zip(dict_encodings_copy, myheadpose_copy)):
                    # print(pose_yaw, pose_yaw_copy)
                    # print(define_pose_category(pose_yaw_copy))
                    # print(np.array(dict_encoding).shape, np.array(dict_encoding_copy).shape)
                    category = define_pose_category(pose_yaw)
                    if category == define_pose_category(pose_yaw_copy):
                        if i == j:
                            if k != l:
                                if category == 'category_1':
                                    # auto_dist_1.append(eucledian_distance(dict_encoding, dict_encoding_copy))
                                    auto_dist_1.append(cosine_distance(dict_encoding, dict_encoding_copy))
                                elif category == 'category_2':
                                    # auto_dist_2.append(eucledian_distance(dict_encoding, dict_encoding_copy))
                                    auto_dist_2.append(cosine_distance(dict_encoding, dict_encoding_copy))
                                elif category == 'category_3':
                                    # auto_dist_3.append(eucledian_distance(dict_encoding, dict_encoding_copy))
                                    auto_dist_3.append(cosine_distance(dict_encoding, dict_encoding_copy))
                        else:
                            if category == 'category_1':
                                # cross_dist_1.append(eucledian_distance(dict_encoding, dict_encoding_copy))
                                cross_dist_1.append(cosine_distance(dict_encoding, dict_encoding_copy))
                            elif category == 'category_2':
                                # cross_dist_2.append(eucledian_distance(dict_encoding, dict_encoding_copy))
                                cross_dist_2.append(cosine_distance(dict_encoding, dict_encoding_copy))
                            elif category == 'category_3':
                                # cross_dist_3.append(eucledian_distance(dict_encoding, dict_encoding_copy))
                                cross_dist_3.append(cosine_distance(dict_encoding, dict_encoding_copy))
        
        # print(len(auto_dist), len(cross_dist))
        plt.figure('auto cosine similarity')
        plt.plot(np.arange(len(auto_dist_1)), auto_dist_1, label=name)
        plt.legend(loc="upper left")
        auto_dist_1 = np.asarray(auto_dist_1)

        plt.figure('cross cosine similarity')
        plt.plot(np.arange(len(cross_dist_1)), cross_dist_1, label=name)
        plt.legend(loc="upper left")
        cross_dist_1 = np.asarray(cross_dist_1)

    #     try:
    #         auto_min.append(np.min(auto_dist))
    #         auto_max.append(np.max(auto_dist))
    #         auto_avg.append(np.mean(auto_dist))
    #         auto_var.append(np.var(auto_dist))
    #         auto_std.append(np.std(auto_dist))
    #         auto_stats = [auto_min, auto_max, auto_avg, auto_var, auto_std]
    #         threshold_auto_min.append(auto_min)
    #         threshold_auto_avg.append(auto_avg)

    #         cross_min.append(np.min(cross_dist))
    #         cross_max.append(np.max(cross_dist))
    #         cross_avg.append(np.mean(cross_dist))
    #         cross_var.append(np.var(cross_dist))
    #         cross_std.append(np.std(cross_dist))
    #         cross_stats = [cross_min, cross_max, cross_avg, cross_var, cross_std]
    #         threshold_cross_min.append(cross_min)
    #         threshold_cross_avg.append(cross_avg)

    #         threshold_dict[name] = np.asarray([auto_stats, cross_stats], dtype=np.float32)

    #     except ValueError: #raised if auto_dist and cross_dist are empty.
    #         pass

    plt.show()
    # if np.max(threshold_auto_min) < np.min(threshold_cross_min): # safest selection
    #     threshold_face = max(threshold_auto_min)
    #     status = 'Best'
    # else:
    #     if np.mean(threshold_auto_avg) < np.min(threshold_cross_avg):
    #         threshold_face = np.mean(threshold_auto_avg)
    #         status = 'Good'
    #     elif np.mean(threshold_auto_avg) < np.mean(threshold_cross_avg):
    #         threshold_face = np.mean(threshold_auto_avg)
    #         status = 'Fine'
    #     else:
    #         threshold_face = np.min(threshold_cross_avg)    #risky selection among above criterions
    #         status = 'Risky'
    
    # print('\n')
    # print('Min of auto_avg: {}'.format(np.min(threshold_auto_avg)))
    # print('Max of auto_avg: {}'.format(np.max(threshold_auto_avg))) # NO USE
    # print('Min of cross_avg: {}'.format(np.min(threshold_cross_avg)))
    # print('Max of cross_avg: {}'.format(np.max(threshold_cross_avg)))
    # print('Avg of auto_avg: {}'.format(np.mean(threshold_auto_avg)))
    # print('Avg of cross_avg: {}'.format(np.mean(threshold_cross_avg)))
    # print('\n')
    # print('Min of auto_min: {}'.format(np.min(threshold_auto_min))) # NO USE
    # print('Max of auto_min: {}'.format(np.max(threshold_auto_min)))
    # print('Avg of auto_min: {}'.format(np.mean(threshold_auto_min)))
    # print('Min of cross_min: {}'.format(np.min(threshold_cross_min)))
    # print('Max of cross_min: {}'.format(np.max(threshold_cross_min))) # NO USE
    # print('Avg of cross_min: {}'.format(np.mean(threshold_cross_min)))
    # print('\n')
    # print('Threshold_face: {} ({})'.format(threshold_face, status))
    # print('\n')

    # file = open("face_threshold.txt","w") 
    # file.write('Min of auto_avg: {} \n'.format(np.min(threshold_auto_avg)))
    # file.write('Max of auto_avg: {} \n'.format(np.max(threshold_auto_avg))) # NO USE
    # file.write('Min of cross_avg: {} \n'.format(np.min(threshold_cross_avg)))
    # file.write('Max of cross_avg: {} \n'.format(np.max(threshold_cross_avg)))
    # file.write('Avg of auto_avg: {} \n'.format(np.mean(threshold_auto_avg)))
    # file.write('Avg of cross_avg: {} \n'.format(np.mean(threshold_cross_avg)))
    # file.write('\n')
    # file.write('Min of auto_min: {} \n'.format(np.min(threshold_auto_min))) # NO USE
    # file.write('Max of auto_min: {} \n'.format(np.max(threshold_auto_min)))
    # file.write('Avg of auto_min: {} \n'.format(np.mean(threshold_auto_min)))
    # file.write('Min of cross_min: {} \n'.format(np.min(threshold_cross_min)))
    # file.write('Max of cross_min: {} \n'.format(np.max(threshold_cross_min))) # NO USE
    # file.write('Avg of cross_min: {} \n'.format(np.mean(threshold_cross_min)))
    # file.write('\n')
    # file.write('Threshold_face: {} ({})'.format(threshold_face, status))
    # file.close()

    # # # name_list, threshold_stats = list(), list()
    # # # 0->min, 1->max, 2->avg, 3->variance, 4->standard deviation
    # # for name, details in threshold_dict.items():
    # #     # print(name)
    # #     threshold_stats = np.squeeze(details, axis=2).tolist()
    # #     auto_stats = threshold_stats[0]
    # #     cross_stats = threshold_stats[1]
    # #     # print(auto_stats)
    # #     # print(cross_stats)
    
def main():
    # cam = cv2.VideoCapture(0)
    # frame_width = int(cam.get(3))
    # frame_height = int(cam.get(4))

    # print(frame_width, frame_height)

    # FIND STATS OF THESHOLD-ELIGIBLE PERSON-FOLDERS
    # print('Tunning face recognition threshold value ...')
    find_threshold_stats()

    # TEST_DATA_SEPARATION
    # test_data_split()

    # TODO: read the recent dictinary and add along with it
    if ADD_FROM_WEBCAM:

        print('There should be only one person at a time in the frame for capturing image.')
        name = input("Enter your name: ")

        while True:
            ret, frame = cam.read()
            frame_show = frame.copy()

            if ret == False:
                break

            frame_show, faces = detect_face(frame, count)

            cv2.putText(frame_show, 'Hi! {}, press q to save your image.'.format(name), 
                (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 1)
            
            # Display the resulting image
            cv2.imshow('Camera', frame_show)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                # cv2.imwrite('known_persons/'+name+'.jpg', frame)
                # serialize_dict('data/known_persons_facenet.pkl', name, mydict)
                break

        # Release handle to the webcam
        cam.release()
        cv2.destroyAllWindows()
    else:
        mydict = dict()
        count = 0
        print('Serializing the face embeddings ... ')
        for dir_name in tqdm(os.listdir(src_path)):
            cropped_faces = list()
            faces_embeddings = list()
            facialLandmarks = list()
            for file in glob.glob(os.path.join(src_path, dir_name)+'/*'):
                # name = file.split('/')[-1]
                frame = cv2.imread(file)
                face_boxes, cropped = detect_face(frame, dir_name, count)
                if cropped is not None:
                    cropped_faces.append(cropped)

                # FIND THE FACIAL LANDMARKS
                frame_landmarks, dlibLandmarks = find_landmarks(frame, [face_boxes])
                facialLandmarks.append(dlibLandmarks)
            
            if len(cropped_faces) > 0:
                faces_embeddings.append(get_facial_embeddings(frame, myfacenet, cropped_faces, facialLandmarks))
                mydict[dir_name] = np.asarray(faces_embeddings, dtype=np.float32)
                # count += 1

        serialize_dict('data/faces_database_128D.pkl', mydict)

if __name__ == '__main__':
    main()
    # Deserialize the dictonary
    # unserialized_dict = deserialize_dict('data/known_persons_facenet.pkl')
    # for key, values in unserialized_dict.items():
    #     # print(key)
    #     # print(np.squeeze(values, axis=0).shape)
    #     for value in values:
    #         for v in value:
    #             print(v)