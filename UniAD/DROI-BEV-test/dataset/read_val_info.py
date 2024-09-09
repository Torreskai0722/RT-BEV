import pickle
import matplotlib.pyplot as plt
import numpy as np

file_path = '/media/hydrapc/hdd-drive3/UniAD-data/infos/nuscenes_infos_temporal_val.pkl'

# file_path = '/home/hydrapc/Downloads/v1.0-trainval_meta/v1.0-trainval/sample_data.json'

with open(file_path, 'rb') as file:
    data = pickle.load(file)

# Define the list of cameras based on your dataset structure
cameras = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']

# List of keys to compare
# keys_to_compare = ['data_path', 'type', 'sample_data_token', 'sensor2ego_translation', 
#                    'sensor2ego_rotation', 'ego2global_translation', 'ego2global_rotation', 
#                    'timestamp', 'sensor2lidar_rotation', 'sensor2lidar_translation', 'cam_intrinsic']

# keys_to_compare = ['ego2global_translation', 'ego2global_rotation', 
#                    'sensor2lidar_rotation', 'sensor2lidar_translation']

print(data['infos'][0].keys())
# dict_keys(['lidar_path', 'token', 'prev', 'next', 'can_bus', 'frame_idx', 
#            'sweeps', 'cams', 'scene_token', 'lidar2ego_translation', 
#            'lidar2ego_rotation', 'ego2global_translation', 
#            'ego2global_rotation', 'timestamp', 'gt_boxes', 
#            'gt_names', 'gt_velocity', 'num_lidar_pts', 'num_radar_pts', 
#            'valid_flag', 'gt_inds', 'gt_ins_tokens', 'fut_traj', 
#            'fut_traj_valid_mask', 'visibility_tokens'])

print(data['infos'][1]['sweeps'][0]['sensor2lidar_translation'])
# for j in range(len(data['infos'])):
#     print(data['infos'][j]['sweeps'])

def check_time_diff():
    keys_to_compare = ['timestamp']
    time_diff = []
    # Compare the specified keys for each camera in the first 'number_of_samples' samples
    for i in range(len(data['infos'])):  # Dynamically use the 'number_of_samples'
        # print(f"\nSample {i+1}:")
        t = []
        # print(data['infos'][i]['sweeps'])
        for cam in cameras:
            # print(f"{cam}:")
            if cam in data['infos'][i]['cams']:
                cam_data = data['infos'][i]['cams'][cam]
                for key in keys_to_compare:
                    # print(cam_data.get(key, 'Key not available')/1e6)
                    t.append(cam_data.get(key, 'Key not available')/1e6)
                    # print(f"{key}: {cam_data.get(key, 'Key not available')}")
            else:
                print(f"Data for {cam} not available in this sample.")
        # t = [ti / 10e6 for ti in t]
        # print(min(t), max(t), max(t)-min(t))
        # print(1000*(max(t)-min(t)))
        time_diff.append(1000*(max(t)-min(t)))

    # Generate a list of indices for the x-axis
    # time_indices = list(range(len(time_diff)))

    # # Create a line plot
    # plt.figure(figsize=(10, 5))  # Set the figure size
    # plt.plot(time_indices, time_diff, marker='o', linestyle='-', color='b')  # Line plot
    # plt.title('Time Differences Between Synced Cameras on nuscenes dataset')  # Title of the plot
    # plt.xlabel('Sample Index')  # Label for the x-axis
    # plt.ylabel('Time Difference (milliseconds)')  # Label for the y-axis
    # plt.grid(True)  # Show grid
    # plt.show()  # Display the plot


def check_cam_intrisics():
    # Initialize a dictionary to store the groups of sample IDs for each unique intrinsic parameter of each camera
    cam_intrinsics_groups = {cam: {} for cam in cameras}
    lidar_cs = {}

    for i in range(len(data['infos'])):  # Dynamically use the 'number_of_samples'
        # print(f"\nSample {i+1}:")
        sample_id = i+1
        t = []
        # print(data['infos'][i]['cams'])
        # print(data['infos'][i].keys())
        # print(data['infos'][i]['lidar2ego_translation'], data['infos'][i]['lidar2ego_rotation'])
        lidar_trans_array = [data['infos'][i]['lidar2ego_translation'], data['infos'][i]['lidar2ego_rotation']]
        lidar_trans_tuple = tuple(map(tuple, lidar_trans_array))
        if lidar_trans_tuple not in lidar_cs:
            lidar_cs[lidar_trans_tuple] = []
        lidar_cs[lidar_trans_tuple].append(sample_id)
        for cam in cameras:
            # print(f"{cam}:")
            # print(data['infos'][i]['cams'][cam]['cam_intrinsic'])
            intrinsic_array = data['infos'][i]['cams'][cam]['cam_intrinsic']
            # Convert the numpy array to a tuple of tuples
            intrinsic_tuple = tuple(map(tuple, intrinsic_array))

            # Check if the intrinsic tuple already has an entry, if not create one
            if intrinsic_tuple not in cam_intrinsics_groups[cam]:
                cam_intrinsics_groups[cam][intrinsic_tuple] = []
            
            # Append the sample ID to the list of sample IDs that share this intrinsic tuple
            cam_intrinsics_groups[cam][intrinsic_tuple].append(sample_id)
            # sample_token = sample['data'][cam]
            # sample_data = self.nusc.get('sample_data', sample_token)
            # sd_record = self.nusc.get('sample_data', sample_token)
            # cs_record = self.nusc.get('calibrated_sensor',
            #                     sd_record['calibrated_sensor_token'])
            # sensor_record = self.nusc.get('sensor', cs_record['sensor_token'])
            # cam_intrinsic = np.array(cs_record['camera_intrinsic'])
            # imsize = (sd_record['width'], sd_record['height'])

            # lidar_cs_record = self.nusc.get('calibrated_sensor', self.nusc.get(
            #     'sample_data', sample['data']['LIDAR_TOP'])['calibrated_sensor_token'])

    # Output the grouped sample IDs for each camera
    # for cam, intrinsics_dict in cam_intrinsics_groups.items():
    #     print(cam)
    #     for intrinsic, sample_ids in intrinsics_dict.items():
    #         print(f"Intrinsic parameters: {np.array(intrinsic)}")
    #         print(f"Sample IDs: {sample_ids}")

    for lidar_trans, sample_ids in lidar_cs.items():
        print(f"LiDAR Trans: {np.array(lidar_trans, dtype=object)}")
        print(f"Sample IDs: {sample_ids}")

# check_cam_intrisics()