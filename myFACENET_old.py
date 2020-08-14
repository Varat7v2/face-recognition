import tensorflow as tf
import cv2
import numpy as np
from facenet import facenet, detect_face
import os, sys, glob
import time
import pickle

modeldir = './model/facenet-20180408-102900.pb'     #output: 512D vector
# modeldir = './model/facenet-20170511-185253.pb'   #output: 128D vector
mtcnn_models='./mtcnn-models'

cropped = []
scaled = []
scaled_reshape = []

def FACENET_EMBEDDINGS(faces):

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
    # CASW - 1
    #         pnet, rnet, onet = detect_face.create_mtcnn(sess, mtcnn_models)

    #         minsize = 20  # minimum size of face
    #         threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    #         factor = 0.709  # scale factor
    #         margin = 44
    #         frame_interval = 3
    #         batch_size = 1000
    #         image_size = 182
    #         input_image_size = 160

    # #         # print('Loading Modal')
    #         facenet.load_model(modeldir)
    #         images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    #         embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    #         phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    #         embedding_size = embeddings.get_shape()[1]

    #         bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
    #         nrof_faces = bounding_boxes.shape[0]

    #         facenet_embeddings = list()
    #         scaled = []
    #         scaled_reshape = []
            
    #         if nrof_faces > 0:
    #             for i, box in enumerate(np.asarray(bounding_boxes, dtype=np.int32)):
    #                 scaled.append(frame[box[1]:box[3], box[0]:box[2]])
    #                 cv2.imwrite(str(i)+'.jpg', scaled[i])
    #                 # scaled.append(facenet.flip(cropped[i], False))
    #                 # scaled.append(cv2.resize(cropped[i], (image_size, image_size), interpolation=cv2.INTER_LINEAR))
    #                 scaled[i] = cv2.resize(scaled[i], (input_image_size,input_image_size),interpolation=cv2.INTER_LINEAR)
    #                 scaled[i] = facenet.prewhiten(scaled[i])
    #                 scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))
    #                 feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
    #                 emb_array = sess.run(embeddings, feed_dict=feed_dict)
    #                 facenet_embeddings.append(emb_array)

    # return facenet_embeddings

            # CASE - 2
            # print('Loading Modal')
            facenet.load_model(modeldir)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            emb_array = np.zeros((1, embedding_size))
                
            feed_dict = {images_placeholder: faces, phase_train_placeholder: False}
            emb_array = sess.run(embeddings, feed_dict=feed_dict)
    
    return emb_array
                    

def main():
    faces = list()
    for img in glob.glob('cropped_images'+'/*.jpg'):
        frame = cv2.resize(cv2.imread(img), (160,160), interpolation=cv2.INTER_LINEAR)
        faces.append(facenet.prewhiten(frame))
    
    face_encoding = FACENET_EMBEDDINGS(faces)
    print(face_encoding.shape)

if __name__ == '__main__':
    main()




































# import tensorflow as tf
# import cv2
# import numpy as np
# from facenet import facenet, detect_face
# import os, sys
# import time
# import pickle

# # modeldir = './model/facenet-20180408-102900.pb'     #output: 512D vector
# modeldir = './model/facenet-20170511-185253.pb'   #output: 128D vector
# classifier_filename = './class/classifier.pkl'
# mtcnn_models='./mtcnn-models'

# def FACENET_ENCODINGS(cropped_frame):
#     with tf.Graph().as_default():
#         gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
#         sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
#         with sess.as_default():
#             minsize = 20  # minimum size of face
#             threshold = [0.6, 0.7, 0.7]  # three steps's threshold
#             factor = 0.709  # scale factor
#             margin = 44
#             frame_interval = 3
#             batch_size = 1000
#             image_size = 182
#             input_image_size = 160

#             # print('Loading Modal')
#             facenet.load_model(modeldir)
#             images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
#             embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
#             phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
#             embedding_size = embeddings.get_shape()[1]
            
#             # cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
#             cropped_frame = facenet.flip(cropped_frame, False)
#             # scaled.append(cv2.resize(cropped[i], (image_size, image_size), interpolation=cv2.INTER_LINEAR))
#             cropped_frame = cv2.resize(cropped_frame, (input_image_size,input_image_size), interpolation=cv2.INTER_CUBIC)
#             cropped_frame = facenet.prewhiten(cropped_frame)
#             cropped_frame.reshape((1, input_image_size, input_image_size, 3))
#             feed_dict = {images_placeholder: cropped_frame, phase_train_placeholder: False}
#             emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
#             # print(emb_array.shape)

#     return emb_array    

# frame = cv2.imread('cropped_faces/test1.jpg')
# # print(frame.shape)
# embedding = FACENET_ENCODINGS(frame)
# print(embedding.shape)