import numpy as np
import scipy.io
import os
import pickle
import pdb


def compute_vel(box_vel, det, frame, s_x, s_y, y_min, y_max, x_limit, H, location, V0, video_id):
    """
    det: (x_1, y_1, x_2, y_2)
    """
    vel = [V0]
    for i, v in enumerate(box_vel):        
        c = np.array([(det[i][0] + det[i][2])/2.0 , det[i][3] - y_min])
      
        if i == 0:
            f = frame[i]
        else:
            k = 1.5
            v_x_trans = (((H[0][0]*v[0] + H[0][1] * v[1]) * (H[2][0] * c[0] + H[2][1] * c[1] + H[2][2]) -
                         (H[0][0] * c[0] + H[0][1] * c[1] + H[0][2]) * (H[2][0] * v[0] + H[2][1] * v[1])) /
                        ((H[2][0] * c[0] + H[2][1] * c[1] + H[2][2])**2))
            v_y_trans = (((H[1][0]*v[0] + H[1][1] * v[1]) * (H[2][0] * c[0] + H[2][1] * c[1] + H[2][2]) -
                         (H[1][0] * c[0] + H[1][1] * c[1] + H[1][2]) * (H[2][0] * v[0] + H[2][1] * v[1])) /
                        ((H[2][0] * c[0] + H[2][1] * c[1] + H[2][2])**2))
            
            instant_vel = np.sqrt(sum([(v_x_trans * s_x)**2, (v_y_trans * s_y )**2]))
            t_delta = frame[i] - f
            vi = instant_vel / t_delta * 30 * 9 / 4
            
            if vi<=3.0:
            	vi = 0.0
            	
            if location in ['Loc1','Loc2']:                
            	if det[i][2] > x_limit :
            		if location=='Loc1':
            			s1 = 1.0 
                		s2 = 1.0 * 1.1
                	else:
                		s1 = 0.7 * 1.1
                		s2 = 1.0 * 1.1
            		 
            	elif location=='Loc1':
            			s1 = 1.0 * 1.1
            			s2 = 1.4 * 1.1
            	else:
            		s1 = 1.0 * 1.1
            		s2 = 1.8 * 1.1
            		
            	a = (s2-s1) / (y_max - y_min)
            	b = s1 
            	s = a * c[1] + b 	
                
                
            	if location == 'Loc1':
            		if video_id == '1':
            			if det[i][1] < 800 and det [i][1] > 550 :
            				if det[i][2] > x_limit:
            					vi = vi * 0.9
                v_estimate = (vel[i-1] * i + vi * s ) / (i+1)
            elif location == 'Loc3':
            	if det[i][2] > x_limit :
            	
            		s1 = 0.8
            		s2 = 0.95
            		a = (s2-s1) / (y_max - y_min)
            		b = s1 
            		
            	else:
            		s1 = 1.3
            		s2 = 1.6
            		a = (s2-s1) / (y_max - y_min)
            		b = s1 
            	
            	s = a * c[1] + b
            	v_estimate = (vel[i-1] + vi * s) / 2.0
            else:
            	v_estimate = (vel[i-1] + vi) / 2.0	
            f = frame[i]
            vel.append(v_estimate)
      
            #pdb.set_trace()
    return vel



# (s_x, s_y, limit)
info = {'Loc1_1': {'V0': 75.0, 's_x': 3.2/6.0, 's_y': 1.5, 'y_min': 500.0, 'y_max':1000.0 ,
                 'L0': 0.2, 'f1': 959.0, 'f2': -182.2639},
        'Loc1_2': {'V0': 75.0, 's_x': 3.2/5, 's_y': 4.0, 'y_min': 300.0, 'y_max':1000.0 , 
                 'L0': 0.2, 'f1': 905.53, 'f2': -71.2 },                 
        'Loc2_1': {'V0': 75.0, 's_x': 3.2/5.0, 's_y': 4.5, 'y_min': 100.0, 'y_max':1000.0 , 
                 'L0': 0.2, 'f1': 1092.8, 'f2': -64.47} ,
        'Loc2_2': {'V0': 75.0, 's_x': 3.2/4.0, 's_y': 5.0, 'y_min': 80.0, 'y_max':1000.0 ,
                 'L0': 0.2, 'f1': 928.7943, 'f2': -67.777},
        'Loc2_4': {'V0': 75.0, 's_x': 3.2/5.0, 's_y': 3.5, 'y_min': 60.0, 'y_max':1000.0 , 
                 'L0': 0.2, 'f1': 955.8269, 'f2': -87.5031},                 
        'Loc3_1': {'V0': 10.0, 's_x': 3.3/8.0, 's_y': 2.0, 'y_min': 445.0, 'y_max':1000.0 , 
        	 'L0': 0.2, 'f1': 1002.1, 'f2': -89.7456},
	'Loc4_1': {'V0': 20.0,'s_x': 2.86/6, 's_y': 1.5, 'y_min': 235.0, 'y_max':1080.0 , 'L0': 0.2, 
		 'f1': 1246.4, 'f2': -91.3983 },
	'Loc4_2': {'V0': 20.0,'s_x': 2.86/9.0, 's_y': 1.5, 'y_min': 170.0, 'y_max':1080.0 , 'L0': 0.2, 
		 'f1': 1374.7, 'f2': -97.633}}

if __name__ == '__main__':

    #track_results_dir = '/gleuclid/pirazh/track1/'   #directory for original sort results without Kalman velocity
    #track_results_dir = '/gleuclid/pirazh/track1/kalman_vel/' # derectory for results with kalman estimated velocity
    track_results_dir = '/scratch0/pirazh_f/Nvidia_Challenge/tracking/results/'
    filenames = []
    filenames = os.listdir(track_results_dir)
    
    f = open('track.txt','w')
    video_index = 0
    for fl in filenames:
    	video_index+=1
        location = fl[:4]
        video_id = fl[5]
        
        if location+'_'+video_id in ['Loc1_1']:
        	item = 'Loc1_1'
        elif location+'_'+video_id in ['Loc1_2','Loc1_3','Loc1_4','Loc1_5','Loc1_6','Loc1_7','Loc1_8']:
        	item = 'Loc1_2'
       	elif location+'_'+video_id in ['Loc2_1']:
        	item = 'Loc2_1'
        elif location+'_'+video_id in ['Loc2_2','Loc2_3']:
        	item = 'Loc2_2'
        elif location+'_'+video_id in ['Loc2_4','Loc2_5','Loc2_6','Loc2_7','Loc2_8']:
        	item = 'Loc2_4'
        elif location+'_'+video_id in ['Loc3_1','Loc3_2','Loc3_3','Loc3_4','Loc3_5']:
        	item = 'Loc3_1'
        elif location+'_'+video_id in ['Loc4_1']:
        	item = 'Loc4_1'
        elif location+'_'+video_id in ['Loc4_2','Loc4_3','Loc4_4','Loc4_5']:
        	item = 'Loc4_2'	
        		
        s_x = info[item]['s_x']
        s_y = info[item]['s_y']
        f1 = info[item]['f1']
        f2 = info[item]['f2']
        l0 = info[item]['L0']
        #H = info[location]['H']
        H = [[l0,-l0*(f1/f2),0.0],[0.0,1.0,0.0],[0.0,-(1/f2),1.0]]
        
        x_limit = 1920/2.0
        y_min = info[item]['y_min']
        y_max = info[item]['y_max']
        V0 = info[item]['V0']


        with open(track_results_dir + fl, 'rb') as t_bbox:
            data = pickle.load(t_bbox)
	
	t_bbox.close()
	
        frame = data['frame_num']
        track_ids = data['track_id']
        detections = data['bbox']
        velocities = data['box_velocities']
        score = data['scores']
        
	
        # filter out bounding boxes that is not in the cropped image
        detections_filtered = []
        track_ids_filtered = []
        frame_filtered = []
        velocities_filtered = []
        score_filtered = []
	#pdb.set_trace()
	
        for i in range(len(detections)):
            det = detections[i]
            if location == 'Loc4':
                if det[1] > y_min :
                    detections_filtered.append(det)
                    velocities_filtered.append(velocities[i])
                    track_ids_filtered.append(track_ids[i])
                    frame_filtered.append(frame[i])
                    score_filtered.append(score[i])

            elif location == 'Loc2':
                if det[1] > y_min:
                    detections_filtered.append(det)
                    velocities_filtered.append(velocities[i])
                    track_ids_filtered.append(track_ids[i])
                    frame_filtered.append(frame[i])
                    score_filtered.append(score[i])

            elif location == 'Loc1':
                if det[1] > y_min :
                    detections_filtered.append(det)
                    velocities_filtered.append(velocities[i])
                    track_ids_filtered.append(track_ids[i])
                    frame_filtered.append(frame[i])
                    score_filtered.append(score[i])

            elif location == 'Loc3':
                if det[1] > y_min:
                    detections_filtered.append(det)
                    velocities_filtered.append(velocities[i])
                    track_ids_filtered.append(track_ids[i])
                    frame_filtered.append(frame[i])
                    score_filtered.append(score[i])
	#pdb.set_trace()
        detections_filtered = np.matrix(detections_filtered)
        velocities_filtered = np.matrix(velocities_filtered)
        track_ids_filtered = np.array(track_ids_filtered)
        frame_filtered = np.array(frame_filtered)
        score_filtered = np.array(score_filtered)

        frame_final = []
        track_id_final = []
        score_final = []
        det_final = []
        vel_final = []
        vel_boxes = []

        for tr in np.unique(track_ids_filtered):
            det = detections_filtered[track_ids_filtered == tr]
            det = det.tolist()
            box_vel = velocities_filtered[track_ids_filtered == tr].tolist()
            tr_frame = frame_filtered[track_ids_filtered == tr].tolist()
            box_score = score_filtered[track_ids_filtered == tr].tolist()
            
            vel = compute_vel(box_vel, det, tr_frame, s_x, s_y, y_min, y_max, x_limit, H, location, V0, video_id)

            frame_final.extend(tr_frame)
            track_id_final.extend([tr] * len(det))
            vel_final.extend(vel)
            det_final.extend(det)
            score_final.extend(box_score)

        data_final = {}
        data_final['track_ids'] = track_id_final
        data_final['frame_num'] = frame_final
        data_final['detections'] = det_final
        data_final['velocity'] = vel_final
        data_final['score'] = score_final
	
	
	print ('Done %s' % fl)
        scipy.io.savemat('./velocity_results_linear/{}.mat'.format(fl[:6]), {'track_id': data_final['track_ids'],'frame_num': data_final['frame_num'], 'bbox': data_final['detections'],'velocity': data_final['velocity'], 'score': data_final['score']})
        for i in range(0, len(track_id_final)):
            f.write('{} {} {} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} \n'.format(video_index, frame_final[i] + 1, int(track_id_final[i]),
                                                                                   det_final[i][0], det_final[i][1], det_final[i][2],
                                                                                   det_final[i][3],
                                                                                   vel_final[i], score_final[i]))
        
        print ('Saving %s.mat' % fl)
        #pdb.set_trace()
    f.close() 









