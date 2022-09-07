import bpy
import math
import csv
from datetime import datetime
from pathlib import Path

PHONE_POINT_LABELS = [#HEAD_POINT_LABELS = [
#    'jamesHead1:M_1',
#    'jamesHead1:M_2',
#    'jamesHead1:M_3',
#    'jamesHead1:M_4',
#    'jamesHead1:M_5'
    'M_1.001',
    'M_2.001',
    'M_3.001',
    'M_4.001',
    'M_5.001',
]

HEAD_POINT_LABELS = [#PHONE_POINT_LABELS = [
#    'jamesPhone1:M_1',
#    'jamesPhone1:M_2',
#    'jamesPhone1:M_3',
#    'jamesPhone1:M_4',
#    'jamesPhone1:M_5'
    'M_1',
    'M_2',
    'M_3',
    'M_4',
    'M_5',
]
PHONE_CENTRE_POINT_LABEL = PHONE_POINT_LABELS[-1]
PHONE_BONE_POINT_LABELS = PHONE_POINT_LABELS[0:3]
HEAD_BONE_POINT_LABELS = [HEAD_POINT_LABELS[1], HEAD_POINT_LABELS[2], HEAD_POINT_LABELS[4]]
# 82
# 16d

MOTIONS = {
    'POINTING_TRANSLATE_PHONE' : [
        'TOP_CENTRE',
        'TOP_RIGHT',
        'MID_RIGHT',
        'BOTTOM_RIGHT',
        'BOTTOM_CENTRE',
        'BOTTOM_LEFT',
        'MID_LEFT',
        'TOP_LEFT'
    ],
    'POINTING_ROTATE_PHONE' : [
        'TOP_CENTRE',
        'TOP_RIGHT',
        'MID_RIGHT',
        'BOTTOM_RIGHT',
        'BOTTOM_CENTRE',
        'BOTTOM_LEFT',
        'MID_LEFT',
        'TOP_LEFT'
    ],
    'POINTING_ROTATE_HEAD' : [
        'TOP_CENTRE',
        'TOP_RIGHT',
        'MID_RIGHT',
        'BOTTOM_RIGHT',
        'BOTTOM_CENTRE',
        'BOTTOM_LEFT',
        'MID_LEFT',
        'TOP_LEFT'
    ],
    'TRANSLATE_PHONE' : [
        'TOP_CENTRE',
        'MID_RIGHT',
        'BOTTOM_CENTRE',
        'MID_LEFT'
    ],
    'CIRCULAR_PHONE' : [
        'CLOCKWISE',
        'ANTI_CLOCKWISE'
    ],
    'CIRCULAR_HEAD' : [
        'CLOCKWISE',
        'ANTI_CLOCKWISE'
    ],
    'ZOOM_PHONE' : [
        'ZOOM_IN',
        'ZOOM_OUT'
    ],
    'ZOOM_HEAD' : [
        'ZOOM_IN',
        'ZOOM_OUT'
    ],
    'ROTATE_HEAD' : [
        'TOP_CENTRE',
        'MID_RIGHT',
        'BOTTOM_CENTRE',
        'MID_LEFT'
    ],
    'ROTATE_PHONE_ROLL' : [
        'CLOCKWISE',
        'ANTI_CLOCKWISE'
    ],
    'ROTATE_HEAD_ROLL' : [
        'CLOCKWISE',
        'ANTI_CLOCKWISE'
    ]
}

frame_rate = 60
frame_Hz = frame_rate / 1000
frame_ms = 1000 / frame_rate
distance_scale = 100 # units converted to cm when exported to fbx...
force_scale = 9.8 # gravity
phone_tracker_mount_compensation = 22.6 # degrees, 0.39 Rads # re-calc this

# need to account for direction participant is facing
# realistically want to track position based on 

D = bpy.data
C = bpy.context

class DataSynchroniser:
    def __init__(self):
        self.frame_end = 0
        self.frame_start = 0
        self.frame_count = 0

    def get_frame_number(self, idx):
        return [int(D.objects[point].animation_data.action.fcurves[0].keyframe_points[idx].co[0]) for point in HEAD_POINT_LABELS + PHONE_POINT_LABELS]

    def load_fbx(self, filepath): # must be full path
        bpy.ops.import_scene.fbx(filepath=filepath, automatic_bone_orientation=True)
        self.frame_end = max(self.get_frame_number(-1))
        self.frame_start = min(self.get_frame_number(0))
        self.frame_count = self.frame_end-self.frame_start
        scn = C.scene
        scn.frame_start = self.frame_start
        scn.frame_end = self.frame_end
        
    def unload_fbx(self):
        for o in bpy.context.scene.objects:
            o.select_set(True)
        bpy.ops.object.delete()

    def sync_shakes(self, shake_data, shake_start_est):
    #    Given the shake data from the csv (maybe csv?), and the windows where estimated to happen, try to sync shakes with frames
    #    Can get shake times from csv, then determine number of frames between them, then use first shake estimate (from viewing playback)
    #    Don't have exact lining-up, so want to not check specific frames, but around them, e.g. check estimate frame, along with next expected frame, then search 2-3 frames ahead / behind subsequent shakes to find best match

        #get_acceleration(label, frame)
        
        est_idx = shake_start_est#-frame_start
        for idx in range(est_idx, est_idx + frame_rate*60): # check 1s past where we think shake started
            accel = self.get_acceleration(PHONE_CENTRE_POINT_LABEL, idx)
            shake = self.get_magnitude(shake_data)
    #        print(accel, shake)
            if (accel >= shake):
                return dict(keyframe_point=idx, keyframe_idx=idx)#+frame_start)
        
    def get_magnitude(self, vector):
        return math.sqrt(sum([pow(acc,2) for acc in vector]))

    # How to deal with cases where keyframe missing?
    def get_acceleration(self, label, timeline_frame_num):
    #    Need to check prior 3 frames for each shake to derive accel (2 frames for velocity, then 2 velocities to get accel
    #      Only issue is that sensor shake is aligned with phone axis, not world ref, so may need to convert movement based on phone orientation

    # How to deal with cases where keyframe missing (co[0])?
        frame = timeline_frame_num# - frame_start
    #    point = [[coords.co[1] for coords in curve.keyframe_points[frame-2:frame+1]] for curve in D.objects[label].animation_data.action.fcurves[0:3]]
        point = [[curve.evaluate(idx) for idx in range(frame-2,frame+2)] for curve in D.objects[label].animation_data.action.fcurves[0:3]]

        velocities = [[((pos[idx+1] - pos[idx])/distance_scale)/(1/frame_rate) for idx in [0,1]] for pos in point]

        acceleration = [(vel[1]-vel[0])/(1/frame_rate) for vel in velocities]
        
        mag = self.get_magnitude(acceleration)
    #    print()
    #    print(point)
    #    print([[(pos[idx+1]/distance_scale, pos[idx]/distance_scale, (pos[idx+1] - pos[idx])/distance_scale) for idx in [0,1]] for pos in point])
    #    print(velocities)
    #    print(acceleration)
    #    print(mag)
        
        return mag / force_scale


    # script.get_pose(PHONE_BONE_POINT_LABELS, 380)
    # For 3 points, Get the position (using centre along hypotenuse as location) and orientation (Quarternion or XYZ Euler?)
    # Also get velocity and acceleration?
    # Do we want relative movement rather than actual position, e.g. angular and linear deltas for each frame?
    def get_pose(self, labels, frame):
        current_pose = {}
        
        def get_bone_points(label): 
            return [curve.evaluate(frame) / distance_scale for curve in D.objects[label].animation_data.action.fcurves[0:3]]
            
        bone_points = []
        for label in labels:
            bone_points.append(get_bone_points(label))
        
        first = bone_points[0]
        second = bone_points[1]
        third = bone_points[2]
        
        # mid point between point 1 & 3 for 'position' of bone
        centre = [(first[i] + third[i]) / 2 for i in range (0, 3)]
        
        # point 3 -> 2 give roll and pitch
        # point 2 -> 1 can then give yaw
        local_first = [first[i] - centre[i] for i in range (0, 3)]
#        print(local_first)
        local_second = [second[i] - centre[i] for i in range (0, 3)]
#        print(local_second)
        local_third = [third[i] - centre[i] for i in range (0, 3)]
#        print(local_third)
        v1 = [(local_second[i] - local_third[i]) for i in range (0, 3)]
        v2 = [(local_first[i] - local_second[i]) for i in range (0, 3)]
#        print(v1)
#        print(v2)
        # pitch is toa w/ v1 z & x, 
        # roll is toa w/ v1 y & z,
        # yaw is toa w/ v2 x & y
        
        # -0.0523184
        # pitch = local_second z & x
        pitch = -math.atan2(v1[2], v1[0]) # not sure why minus
        roll = math.atan2(v2[2], v2[1])
        yaw = math.atan2(v2[1], v2[0])
        
#        print(pitch, roll, yaw)
        
    #    print(first, second, third)
    #    print(centre)
        
        # x,y,z,pitch,yaw,roll
        return ([*centre, pitch, roll, yaw], [*first, *second, *third])

    # May need to adjust pose of phone to account for angle with which tracker is mounted.
    def adjust_pose(self, pose, angular_adjustment, linear_adjustment):
        pose[3] = pose[3] + angular_adjustment
        
    def process_data(self, path, attempt_0_frame_est, load=True, attempt_0_override = None):
        path = Path(path)
        if not path.exists():
            raise Exception('Path does not exist: {}'.format(path)) 
            
        if load:
            self.unload_fbx()
            self.load_fbx(str(path / 'Collection_0{}.fbx'.format(path.name)))
        
        attempt_shakes = []
        with open("{}/ShakeDetected.csv".format(path), newline='') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            csv_reader.__next__()
            for row in csv_reader:
                attempt_shakes.append(dict(
                    path="attempt_{}".format(len(attempt_shakes)),
                    shake_data=dict(
                        timestamp=row[0],
                        x=float(row[1]),
                        y=float(row[2]),
                        z=float(row[3])
                    )
                ))
        previous_attempt = {}
        # sync attempts
        root_output_path = path / 'synced_data'
        root_output_path.mkdir(exist_ok = True)
        for attempt in attempt_shakes:
            shake_data = attempt['shake_data']
            attempt_path = path / attempt['path']
            
            if not attempt_path.exists():
                print('Attempt Path does not exist: {}'.format(attempt_path))
                continue
            
            timestamp = int(datetime.strptime('{}000'.format(shake_data['timestamp']), '%Y-%m-%dT%H-%M-%S-%f').timestamp()*1000)
            if previous_attempt:
                # calc estimated frame based on frame number, and time diff between shakes
                # yyyy-MM-ddTHH-mm-ss-SSS
                # '%Y-%d-%dT%H-%M-%S-%f'
                previous_timestamp = previous_attempt['timestamp']
                frame_est = math.trunc(((timestamp-previous_timestamp) * frame_Hz) - 10 + previous_attempt['frame'])
            else:
                frame_est = attempt_0_frame_est
            
            if not previous_attempt and attempt_0_override is not None:
                sync = {'keyframe_point': attempt_0_override}
            else:
                sync = self.sync_shakes([acc/force_scale for acc in [shake_data['x'], shake_data['y'], shake_data['z']]], frame_est)
            print(sync)
            previous_attempt = {
                'timestamp': timestamp,
                'frame': sync['keyframe_point']
            }
            
            for motion, directions in MOTIONS.items():
                motion_path = attempt_path / motion
                # check path exists
                if not motion_path.exists():
                    print('Motion Path does not exist: {}'.format(motion_path))
                    continue
                
                for direction in directions:
                    direction_path = motion_path / direction
                    # check path exists
                    if not direction_path.exists():
                        print('Direction Path does not exist: {}'.format(direction_path))
                        continue
                    image_path = direction_path / 'images'
                    imu_path = direction_path / 'IMU_GYRO_DATA.csv'
    #                esense_path = direction_path / 'ESENSE_IMU_GYRO_DATA.csv'
                    output_path = root_output_path / '{}.csv'.format(str(direction_path.relative_to(path)).replace('\\', '_'))
                    # check paths exist
                    if not image_path.exists() or not imu_path.exists():# or not esense_path.exists():
                        print('Data is missing: {}'.format([(p, p.exists()) for p in [image_path, imu_path]]))#, esense_path]]))
                        continue
                    # check first image, imu, and esense timestamps to check when first 'frame' is for recording
                    # incrementing the frame number, check if any data exists within frame window (frame timestamp + 0.06s / ~16ms)
                    # add lines to link data (with incrementing ID and time delta?)
                    
                    # sync to image frame rate, or 'hold' onto previous value until new one occurs, and sync on fastest data stream?
                    # Start with first image (or realistically which ever data stream has the latest first frame
                    imu_file = open(imu_path, 'r', newline='')
    #                esense_file = open(esense_path, 'r', newline='')
                    output_file = open(output_path, 'w', newline='')
                    try:
                        print("processing: {}, from frame: {}".format(direction_path, sync['keyframe_point'] + math.trunc(timestamp*frame_Hz)))
                        imu_reader = csv.reader(imu_file, delimiter=',')
    #                    esense_reader = csv.reader(esense_file, delimiter=',')

                        image_header = 'RELATIVE_IMAGE_PATH'
                        image_delta_header = [image_header, 'IMAGE_DELTA_MS']
                        imu_header = next(imu_reader)[1:] # remove header
                        imu_delta_header = [*imu_header, *['{}_DELTA'.format(header) for header in imu_header], 'IMU_DELTA_MS']
    #                    esense_header = next(esense_reader)[1:] # remove header
    #                    esense_delta_header = [*esense_header, *['{}_DELTA'.format(header) for header in esense_header], 'ESENSE_DELTA_MS']
                        mocap_points_header = [
                            'HEAD_1_X','HEAD_1_Y','HEAD_1_Z','HEAD_2_X','HEAD_2_Y','HEAD_2_Z','HEAD_3_X','HEAD_3_Y','HEAD_3_Z',
                            'PHONE_1_X','PHONE_1_Y','PHONE_1_Z','PHONE_2_X','PHONE_2_Y','PHONE_2_Z','PHONE_3_X','PHONE_3_Y','PHONE_3_Z'
                        ]
                        mocap_header = [
                            'HEAD_X', 'HEAD_Y', 'HEAD_Z', 'HEAD_PITCH', 'HEAD_ROLL', 'HEAD_YAW', 'HEAD_X_VEL', 'HEAD_Y_VEL', 'HEAD_Z_VEL', 'HEAD_PITCH_VEL', 'HEAD_ROLL_VEL', 'HEAD_YAW_VEL', 'HEAD_X_ACC', 'HEAD_Y_ACC', 'HEAD_Z_ACC', 'HEAD_PITCH_ACC', 'HEAD_ROLL_ACC', 'HEAD_YAW_ACC',
                            'PHONE_X', 'PHONE_Y', 'PHONE_Z', 'PHONE_PITCH', 'PHONE_ROLL', 'PHONE_YAW', 'PHONE_X_VEL', 'PHONE_Y_VEL', 'PHONE_Z_VEL', 'PHONE_PITCH_VEL', 'PHONE_ROLL_VEL', 'PHONE_YAW_VEL', 'PHONE_X_ACC', 'PHONE_Y_ACC', 'PHONE_Z_ACC', 'PHONE_PITCH_ACC', 'PHONE_ROLL_ACC', 'PHONE_YAW_ACC',
                        ]
                        mocap_delta_header = [*mocap_points_header, *mocap_header, *['{}_DELTA'.format(header) for header in mocap_header], 'MOCAP_DELTA_MS']
                        
                        imu_gen = self.csv_generator(imu_reader, timestamp)
    #                    esense_gen = csv_generator(esense_reader, timestamp)
                        image_gen = self.image_file_generator(image_path, timestamp, image_path.relative_to(path))
                        
                        imu_data_backlog = []
                        imu_data = next(imu_gen, None)
    #                    esense_data_backlog = []
    #                    esense_data = next(esense_gen, None)
                        image_data_backlog = []
                        image_data = next(image_gen, None)
                        
                        output_writer = csv.writer(output_file, delimiter=',')
                        output_writer.writerow(['MS_FROM_START', 'MS_DELTA', *image_delta_header, *imu_delta_header, *mocap_delta_header]) # *esense_delta_header, 
                        
                        if imu_data is None or image_data is None:
                            print('Data is missing: imu: {}, image: {}'.format(imu_data, image_data))
                            continue

                        # get 'latest' timestamp from image vs IMU
                        initial_timestamp = max([image_data['ms_from_attempt_start'], imu_data['ms_from_attempt_start']])
                        current_timestamp = initial_timestamp
                        previous_timestamp = current_timestamp
                        data_to_write = dict(
                            ms_from_start=[0,0],
                            image=['NULL', 0],
                            imu=[0 for _ in imu_delta_header],
    #                        esense=[0 for _ in esense_delta_header],
                            mocap=[0 for _ in mocap_delta_header]
                        )
                        # data['ms_from_start'], *data['mocap'], *data['image'], *data['imu'], *data['esense']
                        # sync everything with current
                        if imu_data['ms_from_attempt_start'] < current_timestamp:
                            imu_data_backlog = [imu_data]
                            while imu_data_backlog[0] is not None and imu_data_backlog[0]['ms_from_attempt_start'] < current_timestamp:
                                # need to hold onto prior data
                                imu_data = imu_data_backlog[0]
                                imu_data_backlog = [next(imu_gen, None)]
                        if image_data['ms_from_attempt_start'] < current_timestamp:
                            image_data_backlog = [image_data]
                            while image_data_backlog[0] is not None and image_data_backlog[0]['ms_from_attempt_start'] < current_timestamp:
                                # need to hold onto prior data
                                image_data = image_data_backlog[0]
                                image_data_backlog = [next(image_gen, None)]
    #                    if esense_data['ms_from_attempt_start'] < current_timestamp:
    #                        esense_data_backlog = [esense_data]
    #                        while esense_data_backlog[0]['ms_from_attempt_start'] < current_timestamp:
    #                            # need to hold onto prior data
    #                            esense_data = esense_data_backlog[0]
    #                            esense_data_backlog = [next(esense_gen, None)]
                     
                        # get first timestamp (based on image or IMU, whichever is latest)
                        # iterate data streams till current frame is for 'first' timestamp
                        # if data exists prior to 'first' timestamp, use prior frame
                        # check 'oldest' timestamp not yet processed, add new row, copying prior data for other streams, get next timesatmp for that data stream.
                        # if data within 5ms, sync and save current data for that data stream too?
                        
                        mocap_gen = self.mocap_generator(sync['keyframe_point'], current_timestamp)
                        mocap_data = next(mocap_gen)
    #                    print(imu_data, image_data, mocap_data)

                        if imu_data is None or image_data is None or mocap_data is None:
                            print('Data is missing: imu: {}, image: {}, mocap: {}'.format(imu_data, image_data, mocap_data))
                            continue

                        if image_data['ms_from_attempt_start'] <= current_timestamp:
                            data_to_write['image'] = image_data['data']
                            image_data = image_data_backlog[0] if len(image_data_backlog) > 0 else next(image_gen, None)
                            
                        if imu_data['ms_from_attempt_start'] <= current_timestamp:
                            data_to_write['imu'] = imu_data['data']
                            imu_data = imu_data_backlog[0] if len(imu_data_backlog) > 0 else next(imu_gen, None)
                            
    #                    if esense_data['ms_from_attempt_start'] <= current_timestamp:
    #                        data_to_write['esense'] = esense_data['data']
    #                        esense_data = esense_data_backlog[0] if len(esense_data_backlog) > 0 else next(esense_gen, None)
                            
                        if mocap_data['ms_from_attempt_start'] <= current_timestamp:
                            data_to_write['mocap'] = mocap_data['data']
                            mocap_data = next(mocap_gen, None)
                            
                        self.write_data(output_writer, data_to_write)

                        while (image_data is not None and imu_data is not None and mocap_data is not None): #and esense_data is not None 
                            current_timestamp = min([image_data['ms_from_attempt_start'], imu_data['ms_from_attempt_start'], mocap_data['ms_from_attempt_start']]) # , esense_data['ms_from_attempt_start']
                            data_to_write['ms_from_start'] = [current_timestamp-initial_timestamp, current_timestamp-previous_timestamp]
                            previous_timestamp = current_timestamp
                            
                            # for each data, check if timestamp less than current (or equal), if so add to data_to_write, and update with either backlog, or next
                            # if gen returns None, data_exists = False, continue
                            
                            if image_data['ms_from_attempt_start'] <= current_timestamp:
                                data_to_write['image'] = image_data['data']
                                image_data = next(image_gen, None)
                            else:
                                data_to_write['image'] = [data_to_write['image'][0], 0]
                                
                            if imu_data['ms_from_attempt_start'] <= current_timestamp:
                                data_to_write['imu'] = imu_data['data']
                                imu_data = next(imu_gen, None)
                            else:
                                data_to_write['imu'] = [*data_to_write['imu'][:len(imu_header)], *[0 for _ in range(len(imu_delta_header)-len(imu_header))]]
                                
    #                        if esense_data['ms_from_attempt_start'] <= current_timestamp:
    #                            data_to_write['esense'] = esense_data['data']
    #                            esense_data = next(esense_gen, None)
    #                        else:
    #                            data_to_write['esense'] = [*data_to_write['esense'][:len(esense_header)], *[0 for _ in range(len(esense_delta_header)-len(esense_header))]]
                                
                            if mocap_data['ms_from_attempt_start'] <= current_timestamp:
                                data_to_write['mocap'] = mocap_data['data']
                                mocap_data = next(mocap_gen, None)
                            else:
                                data_to_write['mocap'] = [*data_to_write['mocap'][:len(mocap_header + mocap_points_header)], *[0 for _ in range(len(mocap_delta_header)-len(mocap_header + mocap_points_header))]]
                            
                            self.write_data(output_writer, data_to_write)
                     
                    except FileNotFoundError as ex:
                        print("FileNotFound: ", ex)
    #                except Exception as ex:
    #                    print("Exception: ", ex)
                    finally:
                        imu_file.close()
    #                    esense_file.close()
                        output_file.close()

    def write_data(self, csv_writer, data):
        csv_writer.writerow([*data['ms_from_start'], *data['image'], *data['imu'], *data['mocap']])#, *data['esense']

    def get_delta(self, current_data, previous_data):
        return [p_1 - p_2 for p_1, p_2 in zip(current_data, previous_data)] if len(previous_data) > 0 else current_data

    def mocap_generator(self, attempt_start_frame, ms_from_attempt_start):
        def get_derivation(current_data, previous_data):
            return [datum/(1/frame_rate) for datum in self.get_delta(current_data, previous_data)]
        
        idx = attempt_start_frame + math.trunc(ms_from_attempt_start*frame_Hz)# - frame_start
        
        head_pose_1 = self.get_pose(HEAD_BONE_POINT_LABELS, idx-1)[0]
        head_pose_2 = self.get_pose(HEAD_BONE_POINT_LABELS, idx-2)[0]
        phone_pose_1 = self.get_pose(PHONE_BONE_POINT_LABELS, idx-1)[0]
        phone_pose_2 = self.get_pose(PHONE_BONE_POINT_LABELS, idx-2)[0]
        
        previous_head = head_pose_1
        previous_phone = phone_pose_1
        previous_head_velocity = get_derivation(head_pose_1, head_pose_2)
        previous_phone_velocity = get_derivation(phone_pose_1, phone_pose_2)
        previous_head_acceleration = get_derivation(previous_head_velocity, get_derivation(head_pose_2, self.get_pose(HEAD_BONE_POINT_LABELS, idx-3)[0]))
        previous_phone_acceleration = get_derivation(previous_phone_velocity, get_derivation(phone_pose_2, self.get_pose(PHONE_BONE_POINT_LABELS, idx-3)[0]))
        
        previous_head_data = []
        previous_phone_data = []
        while idx < self.frame_end:
            # get each item
            (head, head_bones) = self.get_pose(HEAD_BONE_POINT_LABELS, idx)
            head_velocity = get_derivation(head, previous_head)
            previous_head = head
            head_acceleration = get_derivation(head_velocity, previous_head_velocity)
            previous_head_velocity = head_velocity
            
            (phone, phone_bones) = self.get_pose(PHONE_BONE_POINT_LABELS, idx)
            phone_velocity = get_derivation(phone, previous_phone)

            phone_acceleration = get_derivation(phone_velocity, previous_phone_velocity)
            previous_phone_velocity = phone_velocity
            
            head_data=[*head, *head_velocity, *head_acceleration]
            phone_data=[*phone, *phone_velocity, *phone_acceleration]
            head_deltas=self.get_delta(head_data, previous_head_data)
            phone_deltas=self.get_delta(phone_data, previous_phone_data)
            previous_head_data=head_data
            previous_phone_data=phone_data
            
            yield dict(
    #            ms_from_attempt_start=((idx+frame_start-attempt_start_frame) / frame_Hz),
                ms_from_attempt_start=((idx-attempt_start_frame) / frame_Hz),
                data=[*head_bones, *phone_bones, *head_data, *phone_data, *head_deltas, *phone_deltas, frame_ms]
            )
            idx+=1

    def csv_generator(self, reader, attempt_start):
        # Want to track last accel, so can store delta
        previous_data = []
        previous_timestamp = -1
        for row in reader:
            data = [float(datum) for datum in row[1:]]
            deltas = self.get_delta(data, previous_data)
            previous_data = data
            ms_from_attempt_start=int((datetime.strptime('{}000'.format(row[0]), '%Y-%m-%dT%H-%M-%S-%f').timestamp()*1000)-attempt_start)
            ms_delta = ms_from_attempt_start-previous_timestamp if previous_timestamp > -1 else 0
            previous_timestamp = ms_from_attempt_start
            yield dict(
                ms_from_attempt_start=ms_from_attempt_start,
                data=[*data, *deltas, ms_delta]
            )
        
    def image_file_generator(self, path, attempt_start, relative_path):
        previous_timestamp = -1
        for image in path.iterdir():
            ms_from_attempt_start=int((datetime.strptime('{}000'.format(image.stem), '%Y-%m-%dT%H-%M-%S-%f').timestamp()*1000)-attempt_start)
            ms_delta = ms_from_attempt_start-previous_timestamp if previous_timestamp > -1 else 0
            previous_timestamp = ms_from_attempt_start
            yield dict(
                ms_from_attempt_start=ms_from_attempt_start,
                data=['./{}/{}'.format(str(relative_path).replace('\\', '/'), image.stem), ms_delta]
            )
    

# Need to get timestamp from frame number
#  Can then process frames for each given recording, outputting a csv which contains actual position (and velocity and accelleration), IMU data, Earable Data, Image File Path
#  Export csv in folder being processed
# given 100Hz sample rate, each 'frame' is 10ms, so we match timestamps relative to frame found for given attempt shake

# Timestamp                 X           Y           Z           (m/s^2)
# 2022-Aug-09T13:23:33.323	-17.695667	-6.628524	7.1837153
# 2022-Aug-09T13:29:35.341	-27.263739	-5.999219	-6.723294
# 2022-Aug-09T13:35:08.720	-18.44889	-5.099885	-9.028566

# final timestamp: 39:59.579
# frames: 59,565
# time: 986,256 ms

# script = D.texts[0].as_module()
processor = DataSynchroniser()
#script.processor.process_data('G:/Study/4', 900)
#script.processor.process_data('G:/Study/0', 3000)
#script.processor.process_data('G:/Study/1', 1000)
#script.processor.process_data('G:/Study/2', -320, attempt_0_override=-320)
#script.processor.process_data('G:/Study/3', 20000)
#script.processor.process_data('G:/Study/7', 1900)
# change labels
#script.processor.process_data('G:/Study/6', 8900)
#script.processor.process_data('G:/Study/5', 9500)
