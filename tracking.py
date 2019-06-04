import pickle
import os
import scipy.io
import utilities
from sort import *
import argparse


def main(args):

    # Specify
    detection_results_path = args.detection_results_path

    for fl in os.listdir(detection_results_path):
   
        video_name = fl[:-4]
        print('Processing the detection file for video {} ...\n'.format(video_name))

        # Load the stored pickle file for the detections
        with open(os.path.join(detection_results_path, fl), 'rb') as f_bbox:
            metadata = pickle.load(f_bbox)

        # Construct the dictionary to contain tracking results
        tracking_dict = {}
        tracking_dict['bbox'], tracking_dict['frame_num'], tracking_dict['track_id'] = [], [], []
        tracking_dict['box_velocities'], tracking_dict['scores'] = [], []

        # reset the track ID of the tracker for new video
        KalmanBoxTracker.count = 0

        # establish the multi object tracker
        mot_tracker = Sort()

        # Processing video frames 1 by 1
        for j, md in enumerate(metadata):
            final_boxes, required_labels = [], []
            cls_bbox, label = utilities.convert_from_cls_format(md['cls_boxes'])

            if cls_bbox is None:
                continue

            # Filtering out only the detections for cars, trucks and buses
            for bb, lab in zip(cls_bbox, label):
                if lab in [3, 6, 8]:
                    final_boxes.append(bb)

            if len(final_boxes) == 0:
                continue

            # Convert the lists to numpy arrays
            final_boxes = np.asarray(final_boxes)

            # Imposing detection confidence and bounding box size to discard faulty boxes
            final_boxes = final_boxes[np.logical_and(final_boxes[:, 4] > args.min_det_score, final_boxes[:, 2] -
                                                     final_boxes[:, 0] < args.max_det_size)]

            # Apply NMS
            indices = utilities.non_max_suppression(final_boxes, args.nms_threshold, final_boxes[:, 4])
            
            if len(indices) == 0:
                continue
            
            final_boxes = [final_boxes[i] for i in indices]

            # Update the tracker by feeding the current frame detected boxes
            track_bbs_ids = mot_tracker.update(np.array(final_boxes))

            if len(track_bbs_ids) == 0:
                continue

            # Writing tracking results to the dictionary
            tracking_dict['bbox'].append(track_bbs_ids[:, :4])
            tracking_dict['box_velocities'].append(track_bbs_ids[:, -3:-1])
            tracking_dict['scores'].extend(track_bbs_ids[:, 4])
            tracking_dict['track_id'].extend(track_bbs_ids[:, 5])
            tracking_dict['frame_num'].extend([j] * len(track_bbs_ids))

        # Stacking the tracking results for convenience
        tracking_dict['bbox'] = np.vstack(tracking_dict['bbox'])
        tracking_dict['box_velocities'] = np.vstack(tracking_dict['box_velocities'])

        # writing the tracking results to the pickle files and Matlab files for subsequent stages and visualization
        with open('./results/track1/' + video_name + '.pkl', 'wb') as f:
            pickle.dump(tracking_dict, f)

        scipy.io.savemat('./results/{}.mat'.format(video_name), {'track_id': tracking_dict['track_id'],
                                                               'frame_num': tracking_dict['frame_num'],
                                                               'bbox': tracking_dict['bbox'],
                                                               'scores': tracking_dict['scores'],
                                                               'velocity': tracking_dict['box_velocities']})


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Running Tracker')
    parser.add_argument('--detection_results_path', default='/scratch0/pirazh_f/Nvidia_Challenge/detectron_results/t1'
                        , help='the path to detection results', required=True, type=str)
    parser.add_argument('--min_det_score', default=0.3, help='Minimum detected objects confidence score', type=float)
    parser.add_argument('--max_det_size', default=600, help='Maximum number of pixels a detected '
                                                            'box can occupy', type=int)
    parser.add_argument('--nms_threshold', default=0.9, help='Non-maximal Suppression score')

    args = parser.parse_args()
    main(args)
