import numpy as np

import pickle
import os
import scipy.io
from detectron_results.load_bbox_sample import *
#from sort_original import *
from sort import *
import argparse
import cv2
from detectron_results.preprocessing import *
import pdb



def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA) * (yB - yA)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


if __name__ == '__main__':
    args = parse_args()
    data_root_dir_txt = '/scratch0/pirazh_f/Nvidia_Challenge/detectron_results/t1'


    for fl in os.listdir(data_root_dir_txt):
   
        filename = fl[:-4]
        print(filename)
        with open(os.path.join(data_root_dir_txt, fl), 'rb') as f_bbox:
            metadata = pickle.load(f_bbox)

        dic = {}
        dic['bbox'] = []
        dic['frame_num'] = []
        dic['track_id'] = []
        dic['box_velocities'] = []
        dic['scores'] = []

        KalmanBoxTracker.count = 0
        mot_tracker = Sort()

        for j, md in enumerate(metadata):
            final_boxes = []
            req_labels = []
            cls_bbox, label = convert_from_cls_format(md['cls_boxes'])

            if cls_bbox is None:
                continue

            # keeping only car truck and bus
            for bb, lab in zip(cls_bbox, label):
                if lab in [3, 6, 8]:
                    final_boxes.append(bb)
                    req_labels.append(lab)

            if len(final_boxes) == 0:
                continue

            final_boxes = np.asarray(final_boxes)
            req_labels = np.asarray(req_labels)
            detections = final_boxes[np.logical_and(final_boxes[:,4]>0.3,final_boxes[:,2]-final_boxes[:,0]<600)]
            req_labels = req_labels[np.logical_and(final_boxes[:,4]>0.3,final_boxes[:,2]-final_boxes[:,0]<600)]

            indices = non_max_suppression(detections, 0.9, detections[:, 4])
            
            if len(indices) == 0:
                continue
            
            detections = [detections[i] for i in indices]
            req_labels = [req_labels[i] for i in indices]

            track_bbs_ids = mot_tracker.update(np.array(detections))

            if len(track_bbs_ids) == 0:
                continue

            dic['bbox'].append(track_bbs_ids[:, :4])
            dic['box_velocities'].append(track_bbs_ids[:, -3:-1])
            dic['scores'].extend(track_bbs_ids[:, 4])
            dic['track_id'].extend(track_bbs_ids[:, 5])
            dic['frame_num'].extend([j] * len(track_bbs_ids))

        dic['bbox'] = np.vstack(dic['bbox'])
        dic['box_velocities'] = np.vstack(dic['box_velocities'])

        with open('./results/track1/' + filename + '.pkl', 'wb') as f:
            pickle.dump(dic, f)

        scipy.io.savemat('./results/{}.mat'.format(filename), {'track_id': dic['track_id'],
                                                               'frame_num': dic['frame_num'],
                                                               'bbox': dic['bbox'],
                                                               'scores': dic['scores'],
                                                               'velocity': dic['box_velocities']})


