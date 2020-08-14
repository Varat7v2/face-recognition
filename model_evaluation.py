import os, glob, sys
import tensorflow as tf
import numpy as np
import cv2
from tqdm import tqdm
import random
import pandas as pd
import pickle
import time

import FaceRecognition as FR
from facenet import facenet, detect_face
from myFACENET import FACENET_EMBEDDINGS

from faceDetection_frozenGraph import TensoflowFaceDector

# PATH_TO_CKPT = 'model/frozen_inference_graph_face.pb'
PATH_TO_CKPT = 'model/myssd_mobilenet_v2_face.pb'
tDetector = TensoflowFaceDector(PATH_TO_CKPT)
SERIALIZE_DICT_FIRST = True

test_dir = 'data/lfw_test'
dst_dir = 'data/processed_images'

if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

def serialize_dict(mydict, filename):
    # Serialize dictionary in binary format
    with open(filename, 'wb') as handle:
        pickle.dump(mydict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def deserialize_dict(filename):
    # Deserialize dictionary
    with open(filename, 'rb') as handle:
        return pickle.load(handle)

def get_model_scores(pred_boxes):
    """Creates a dictionary of from model_scores to image ids.
    Args:
        pred_boxes (dict): dict of dicts of 'boxes' and 'scores'
    Returns:
        dict: keys are model_scores and values are image ids (usually filenames)
    """
    model_score={}
    for img_id, val in pred_boxes.items():
        for score in val['scores']:
            if score not in model_score.keys():
                model_score[score]=[img_id]
            else:
                model_score[score].append(img_id)
    return model_score


def calc_iou( gt_bbox, pred_bbox):
    '''
    This function takes the predicted bounding box and ground truth bounding box and 
    return the IoU ratio
    '''
    x_topleft_gt, y_topleft_gt, x_bottomright_gt, y_bottomright_gt= gt_bbox
    x_topleft_p, y_topleft_p, x_bottomright_p, y_bottomright_p= pred_bbox
    
    if (x_topleft_gt > x_bottomright_gt) or (y_topleft_gt> y_bottomright_gt):
        raise AssertionError("Ground Truth Bounding Box is not correct")
    if (x_topleft_p > x_bottomright_p) or (y_topleft_p> y_bottomright_p):
        raise AssertionError("Predicted Bounding Box is not correct")
        
         
    #if the GT bbox and predcited BBox do not overlap then iou=0
    if(x_bottomright_gt < x_topleft_p):
        # If bottom right of x-coordinate  GT  bbox is less than or above the top left of x coordinate of  the predicted BBox
        
        return 0.0
    if(y_bottomright_gt < y_topleft_p):  # If bottom right of y-coordinate  GT  bbox is less than or above the top left of y coordinate of  the predicted BBox
        
        return 0.0
    if(x_topleft_gt > x_bottomright_p): # If bottom right of x-coordinate  GT  bbox is greater than or below the bottom right  of x coordinate of  the predcited BBox
        
        return 0.0
    if(y_topleft_gt > y_bottomright_p): # If bottom right of y-coordinate  GT  bbox is greater than or below the bottom right  of y coordinate of  the predcited BBox
        
        return 0.0
    
    
    GT_bbox_area = (x_bottomright_gt -  x_topleft_gt + 1) * (  y_bottomright_gt -y_topleft_gt + 1)
    Pred_bbox_area =(x_bottomright_p - x_topleft_p + 1 ) * ( y_bottomright_p -y_topleft_p + 1)
    
    x_top_left =np.max([x_topleft_gt, x_topleft_p])
    y_top_left = np.max([y_topleft_gt, y_topleft_p])
    x_bottom_right = np.min([x_bottomright_gt, x_bottomright_p])
    y_bottom_right = np.min([y_bottomright_gt, y_bottomright_p])
    
    intersection_area = (x_bottom_right- x_top_left + 1) * (y_bottom_right-y_top_left  + 1)
    
    union_area = (GT_bbox_area + Pred_bbox_area - intersection_area)
   
    return intersection_area/union_area

def calc_precision_recall(image_results):
    """Calculates precision and recall from the set of images
    Args:
        img_results (dict): dictionary formatted like:
            {
                'img_id1': {'true_pos': int, 'false_pos': int, 'false_neg': int},
                'img_id2': ...
                ...
            }
    Returns:
        tuple: of floats of (precision, recall)
    """
    true_positive=0
    false_positive=0
    false_negative=0
    for img_id, res in image_results.items():
        true_positive += res['true_positive']
        false_positive += res['false_positive']
        false_negative += res['false_negative']
        try:
            precision = true_positive / (true_positive + false_positive)
        except ZeroDivisionError:
            precision = 0.0
        try:
            recall = true_positive / (true_positive + false_negative)
        except ZeroDivisionError:
            recall = 0.0
    return precision, recall

def get_single_image_results(gt_boxes, pred_boxes, iou_thr=0.5):
    """Calculates number of true_pos, false_pos, false_neg from single batch of boxes.
    Args:
        gt_boxes (list of list of floats): list of locations of ground truth
            objects as [xmin, ymin, xmax, ymax]
        pred_boxes (dict): dict of dicts of 'boxes' (formatted like `gt_boxes`)
            and 'scores'
        iou_thr (float): value of IoU to consider as threshold for a
            true prediction.
    Returns:
        dict: true positives (int), false positives (int), false negatives (int)
    """
    all_pred_indices= range(len(pred_boxes))
    all_gt_indices=range(len(gt_boxes))
    if len(all_pred_indices)==0:
        tp=0
        fp=0
        fn=0
        return {'true_positive':tp, 'false_positive':fp, 'false_negative':fn}
    if len(all_gt_indices)==0:
        tp=0
        fp=0
        fn=0
        return {'true_positive':tp, 'false_positive':fp, 'false_negative':fn}
    
    gt_idx_thr=[]
    pred_idx_thr=[]
    ious=[]
    for ipb, pred_box in enumerate(pred_boxes):
        for igb, gt_box in enumerate(gt_boxes):
            iou= calc_iou(gt_box, pred_box)
            
            if iou >iou_thr:
                gt_idx_thr.append(igb)
                pred_idx_thr.append(ipb)
                ious.append(iou)
    iou_sort = np.argsort(ious)[::1]
    if len(iou_sort)==0:
        tp=0
        fp=0
        fn=0
        return {'true_positive':tp, 'false_positive':fp, 'false_negative':fn}
    else:
        gt_match_idx=[]
        pred_match_idx=[]
        for idx in iou_sort:
            gt_idx=gt_idx_thr[idx]
            pr_idx= pred_idx_thr[idx]
            # If the boxes are unmatched, add them to matches
            if(gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
                gt_match_idx.append(gt_idx)
                pred_match_idx.append(pr_idx)
        tp= len(gt_match_idx)
        fp= len(pred_boxes) - len(pred_match_idx)
        fn = len(gt_boxes) - len(gt_match_idx)
    return {'true_positive': tp, 'false_positive': fp, 'false_negative': fn}

# def detect_face_frozen_graph(frame, name):
#     im_height, im_width = frame.shape[0:2]
#     # MODEL PREDICTIONS
#     boxes, scores, classes, num_detections = tDetector.run(frame)
#     boxes = np.squeeze(boxes)
#     scores = np.squeeze(scores)
#     pred_boxes = list()
    
#     for idx, score, box in zip(range(len(scores)), scores, boxes):
#         if score > 0.3:
#             # ymin, xmin, ymax, xmax = box
#             left = int(box[1]*im_width)
#             top = int(box[0]*im_height)
#             right = int(box[3]*im_width)
#             bottom = int(box[2]*im_height)
#             pred_boxes.append([left, top, right, bottom])

#             # pred_boxes.append(pred_box)
#             # cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 3)
#             # cv2.imwrite(dst_dir +'/'+ name + '.jpg', frame)
#             # cv2.putText(frame, '{}'.format(idx), (right-20, top-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA) 

#     return pred_boxes

def detect_face(frame):
    frame_height, frame_width, _ = frame.shape

    boxes, scores, classes, num_detections = tDetector.run(frame)
    boxes = np.squeeze(boxes)
    scores = np.squeeze(scores)
    faces = list()
    scores_updated = list(score for score in scores if score > 0)
    score_count = len(scores_updated)
    boxes_updated = list(boxes[i] for i in range(score_count))

    if score_count == 1:
        for score, box in zip(scores_updated, boxes_updated):
            if score > 0.6:
                # H, status = cv2.findHomography(pts_src, pts_dst)
                left = int(box[1]*frame_width)
                top = int(box[0]*frame_height)
                right = int(box[3]*frame_width)
                bottom = int(box[2]*frame_height)

                cropped_img = frame[top:bottom, left:right]
                cropped_img = cv2.resize(cropped_img, (160,160), interpolation=cv2.INTER_LINEAR)
                cropped = facenet.prewhiten(cropped_img)
                
                # Saving cropped face
                # if not os.path.exists(os.path.join(dst_path, dir_name)):
                #     os.makedirs(os.path.join(dst_path, dir_name))
                # cv2.imwrite(os.path.join(dst_path, dir_name) + '/' + str(count) + '_test.jpg', cropped_img)

                # cv2.rectangle(frame, (left, top), (right, bottom),(0, 255, 0), int(round(frame_height/150)), 8)
                # TODO: For multi-faces we have to CLUSTERING
            else:
                cropped = None
        
        return cropped #TODO: when no faces detected send NONE; This is only for single face in a image
    else:
        return None

def get_facial_embeddings(myfacenet, faces):
    return myfacenet.run(faces)

def cosine_distance(dict_encoding, current_encoding):
    return (1 - np.dot(dict_encoding, current_encoding)/(np.linalg.norm(dict_encoding)*np.linalg.norm(current_encoding)))
    
def  get_avg_precision_at_iou(gt_boxes, pred_bb, iou_thr=0.5):
    
    model_scores = get_model_scores(pred_bb)
    sorted_model_scores= sorted(model_scores.keys())# Sort the predicted boxes in descending order (lowest scoring boxes first):
    for img_id in pred_bb.keys():
        
        arg_sort = np.argsort(pred_bb[img_id]['scores'])
        pred_bb[img_id]['scores'] = np.array(pred_bb[img_id]['scores'])[arg_sort].tolist()
        pred_bb[img_id]['boxes'] = np.array(pred_bb[img_id]['boxes'])[arg_sort].tolist()
        pred_boxes_pruned = deepcopy(pred_bb)
    
    precisions = []
    recalls = []
    model_thrs = []
    img_results = {}# Loop over model score thresholds and calculate precision, recall
    for ithr, model_score_thr in enumerate(sorted_model_scores[:-1]):
            # On first iteration, define img_results for the first time:
        print("Mode score : ", model_score_thr)
        img_ids = gt_boxes.keys() if ithr == 0 else model_scores[model_score_thr]

        for img_id in img_ids:       
            gt_boxes_img = gt_boxes[img_id]
            box_scores = pred_boxes_pruned[img_id]['scores']
            start_idx = 0
            for score in box_scores:
                if score <= model_score_thr:
                    pred_boxes_pruned[img_id]
                    start_idx += 1
                else:
                    break 
            # Remove boxes, scores of lower than threshold scores:
            pred_boxes_pruned[img_id]['scores']= pred_boxes_pruned[img_id]['scores'][start_idx:]
            pred_boxes_pruned[img_id]['boxes']= pred_boxes_pruned[img_id]['boxes'][start_idx:]# Recalculate image results for this image
            print(img_id)
            img_results[img_id] = get_single_image_results(gt_boxes_img, pred_boxes_pruned[img_id]['boxes'], iou_thr=0.5)# calculate precision and recall
        prec, rec = calc_precision_recall(img_results)
        precisions.append(prec)
        recalls.append(rec)
        model_thrs.append(model_score_thr)
        precisions = np.array(precisions)
    recalls = np.array(recalls)
    prec_at_rec = []
    for recall_level in np.linspace(0.0, 1.0, 11):
        try:
            args= np.argwhere(recalls>recall_level).flatten()
            prec= max(precisions[args])
            print(recalls,"Recall")
            print(recall_level,"Recall Level")
            print(args, "Args")
            print(prec, "precision")
        except ValueError:
            prec=0.0
        prec_at_rec.append(prec)
    avg_prec = np.mean(prec_at_rec) 
    return {
        'avg_precision': avg_prec,
        'precisions': precisions,
        'recalls': recalls,
        'model_thrs': model_thrs}

def main():
    global myfacenet

    PATH_TO_CKPT_FACENET_128D = 'model/facenet-20170511-185253.pb'
    PATH_TO_CKPT_FACENET_512D = 'model/facenet-20180408-102900.pb'
    myfacenet = FACENET_EMBEDDINGS(PATH_TO_CKPT_FACENET_512D)

    gt_dict = deserialize_dict('data/known_persons_facenet.pkl')
    known_face_names, known_face_encodings = list(), list()

    for dir_name in tqdm(os.listdir(test_dir)):
        for test_img in glob.glob(os.path.join(test_dir, dir_name)+'/*'):
            cropped_faces = list()
            frame = cv2.imread(test_img)
            cropped = detect_face(frame)
            if cropped is not None:
                cropped_faces.append(cropped)
                face_embedding = np.asarray(get_facial_embeddings(myfacenet, cropped_faces))
                
                # COMPARE THE EMBEDDINGS TO ALL IN THE DICTIONARY
                for dict_name, dict_embeddings in gt_dict.items():
                    known_face_encodings = np.squeeze(dict_embeddings, axis=0)
                    tp = 0
                    tn = 0
                    fn = 0
                    for gt_face in known_face_encodings:
                        dist = cosine_distance(gt_face, face_embedding)
                        if dist < face_threshold:
                            if dir_name == dict_name:
                                tp += 1
                            else:
                                fp += 1
                        else:
                            fn += 1

    
    sys.exit(0)

    # Calculate precision and recall
    precision, recall = calc_precision_recall(image_results)
    f1_score = 2*(precision*recall)/(precision+recall)
    print('Precision= {:.2f}, Recall={:.2f}'.format(precision*100, recall*100))
    print('F1-score= {:.2f}'.format(f1_score*100))

if __name__ == '__main__':
    main()

# READ IMAGE WITH TENSORFLOW
# Tensorflow Image Decode
# IMG_WIDTH, IMG_HEIGHT = 640, 480
# def decode_img(img):
#   # convert the compressed string to a 3D uint8 tensor
#   img = tf.image.decode_jpeg(img, channels=3)
#   # Use `convert_image_dtype` to convert to floats in the [0,1] range.
#   img = tf.image.convert_image_dtype(img, tf.float32)
#   # resize the image to the desired size.
#   return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

# for dir in os.listdir(test_dir): #(Uncomment for oringinal test datasets)
#     for img_path in tqdm(glob.glob(os.path.join(test_dir,dir)+'/*.jpg')):
#         filenames.append(img_path)
#         # img = tf.io.read_file(img_path)
#         # img = decode_img(tf.io.read_file(img_path))
#         frame = cv2.imread(img_path)

#         im_height, im_width, im_channel = frame.shape
#         # image = cv2.flip(image, 1)