import numpy as np
import os
import pickle
import utilities
import argparse

# Dictionary containing the calculated scales for x & y directions, as well as vanishing points for each location
info = {'Loc1_1': {'V0': 75.0, 's_x': 3.2/6.0, 's_y': 1.5, 'y_min': 500.0, 'y_max': 1000.0,
                    'L0': 0.2, 'f1': 959.0, 'f2': -182.2639},
        'Loc1_2': {'V0': 75.0, 's_x': 3.2/5, 's_y': 4.0, 'y_min': 300.0, 'y_max': 1000.0,
                    'L0': 0.2, 'f1': 905.53, 'f2': -71.2 },
        'Loc2_1': {'V0': 75.0, 's_x': 3.2/5.0, 's_y': 4.5, 'y_min': 100.0, 'y_max': 1000.0,
                    'L0': 0.2, 'f1': 1092.8, 'f2': -64.47} ,
        'Loc2_2': {'V0': 75.0, 's_x': 3.2/4.0, 's_y': 5.0, 'y_min': 80.0, 'y_max': 1000.0,
                    'L0': 0.2, 'f1': 928.7943, 'f2': -67.777},
        'Loc2_4': {'V0': 75.0, 's_x': 3.2/5.0, 's_y': 3.5, 'y_min': 60.0, 'y_max': 1000.0,
                    'L0': 0.2, 'f1': 955.8269, 'f2': -87.5031},
        'Loc3_1': {'V0': 10.0, 's_x': 3.3/8.0, 's_y': 2.0, 'y_min': 445.0, 'y_max': 1000.0,
                   'L0': 0.2, 'f1': 1002.1, 'f2': -89.7456},
        'Loc4_1': {'V0': 20.0, 's_x': 2.86/6, 's_y': 1.5, 'y_min': 235.0, 'y_max': 1080.0,
                   'L0': 0.2, 'f1': 1246.4, 'f2': -91.3983},
        'Loc4_2': {'V0': 20.0, 's_x': 2.86/9.0, 's_y': 1.5, 'y_min': 170.0, 'y_max': 1080.0,
                   'L0': 0.2, 'f1': 1374.7, 'f2': -97.633}}


def main(args):

    filenames = os.listdir(args.track_results_path)

    # Write the system output to a txt file in the format acceptable by evaluation server
    f = open(args.output_submission_file, 'w')
    video_index = 0
    for fl in filenames:
        print('Processing the video file {} ... \n'.format(fl[:6] + '.mp4'))
        video_index += 1
        
        if fl[:6] in ['Loc1_1']:
            item = 'Loc1_1'
        elif fl[:6] in ['Loc1_2', 'Loc1_3', 'Loc1_4', 'Loc1_5', 'Loc1_6', 'Loc1_7', 'Loc1_8']:
            item = 'Loc1_2'
        elif fl[:6] in ['Loc2_1']:
            item = 'Loc2_1'
        elif fl[:6] in ['Loc2_2', 'Loc2_3']:
            item = 'Loc2_2'
        elif fl[:6] in ['Loc2_4', 'Loc2_5', 'Loc2_6', 'Loc2_7', 'Loc2_8']:
            item = 'Loc2_4'
        elif fl[:6] in ['Loc3_1', 'Loc3_2', 'Loc3_3', 'Loc3_4', 'Loc3_5']:
            item = 'Loc3_1'
        elif fl[:6] in ['Loc4_1']:
            item = 'Loc4_1'
        elif fl[:6] in ['Loc4_2', 'Loc4_3', 'Loc4_4', 'Loc4_5']:
            item = 'Loc4_2'

        s_x, s_y = info[item]['s_x'], info[item]['s_y']
        f1, f2, l0 = info[item]['f1'], info[item]['f2'], info[item]['L0']

        # Specify H matrix to rectify videos of each location
        H = [[l0, -l0*(f1/f2), 0.0], [0.0, 1.0, 0.0], [0.0, -(1/f2), 1.0]]

        y_min = info[item]['y_min']
        y_max = info[item]['y_max']
        V0 = info[item]['V0']

        # Get the tracking results for the specified video
        with open(args.track_results_path + fl, 'rb') as t_bbox:
            data = pickle.load(t_bbox)

        frame, track_ids, detections = data['frame_num'], data['track_id'], data['bbox']
        velocities, score = data['box_velocities'], data['scores']

        # filter out bounding boxes that is not in the cropped image
        detections_filtered, track_ids_filtered, frame_filtered, velocities_filtered, score_filtered = [], [], [], [], []

        # Only Consider the detection and tracks in the region of interest
        for i in range(len(detections)):
            det = detections[i]
            if det[1] > y_min:
                detections_filtered.append(det)
                velocities_filtered.append(velocities[i])
                track_ids_filtered.append(track_ids[i])
                frame_filtered.append(frame[i])
                score_filtered.append(score[i])

        # Convert to numpy arrays
        detections_filtered = np.matrix(detections_filtered)
        velocities_filtered = np.matrix(velocities_filtered)
        track_ids_filtered = np.array(track_ids_filtered)
        frame_filtered = np.array(frame_filtered)
        score_filtered = np.array(score_filtered)

        frame_final, track_id_final, score_final, det_final, vel_final, vel_boxes = [], [], [], [], [], []

        # Considering all detections corresponding to a given track id to calculate its speed
        for tr in np.unique(track_ids_filtered):
            det = detections_filtered[track_ids_filtered == tr]
            det = det.tolist()
            box_vel = velocities_filtered[track_ids_filtered == tr].tolist()
            tr_frame = frame_filtered[track_ids_filtered == tr].tolist()
            box_score = score_filtered[track_ids_filtered == tr].tolist()

            # measure the velocity
            vel = utilities.compute_vel(box_vel, det, tr_frame, s_x, s_y, y_min, y_max, H, V0)

            frame_final.extend(tr_frame)
            track_id_final.extend([tr] * len(det))
            vel_final.extend(vel)
            det_final.extend(det)
            score_final.extend(box_score)

        # Writing Final Results to a dictionary
        data_final = {}
        data_final['track_ids'] = track_id_final
        data_final['frame_num'] = frame_final
        data_final['detections'] = det_final
        data_final['velocity'] = vel_final
        data_final['score'] = score_final

        # writing files to the submission file
        for i in range(0, len(track_id_final)):
            f.write('{} {} {} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} \n'.format(video_index, frame_final[i] + 1,
                                                                                   int(track_id_final[i]),
                                                                                   det_final[i][0], det_final[i][1],
                                                                                   det_final[i][2],
                                                                                   det_final[i][3],
                                                                                   vel_final[i], score_final[i]))
    f.close() 


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Nvidia AI City 2018 Vehicle Speed Estimation')
    parser.add_argument('--tracking_results_path', default='./tracking_results/',
                        help='The path to the tracking results')
    parser.add_argument('--output_submission_file', default='./submission_result/track.txt')
    args = parser.parse_args()
    main(args)





