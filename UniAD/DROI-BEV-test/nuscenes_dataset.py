from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
import pickle
import numpy as np

# nusc = NuScenes(version='v1.0-trainval', dataroot='/media/hydrapc/hdd-drive3/UniAD-data/nuscenes', verbose=True)

nusc = NuScenes(version='v1.0-mini', dataroot='/home/hydrapc/Downloads/v1.0-mini', verbose=True)
nusc_can_bus = NuScenesCanBus(dataroot='/media/hydrapc/hdd-drive3/UniAD-data/nuscenes')
# nusc.list_scenes()
# nusc.list_sample()

# scene = nusc.scene[0]
# sample_token = scene['first_sample_token']
# sample = nusc.get('sample', sample_token)
# print(sample['data'])
cams = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']

# token = sample['data']['CAM_FRONT']
# while token != '':
#     data = nusc.get('sample_data', token)
#     # print(data)
#     print("CAM_FRONT",data["is_key_frame"],data["timestamp"], data['filename'])
#     token = data["next"]

# Initialize a dictionary to store all camera data
all_camera_data = {cam: [] for cam in cams}

# Function to save data to disk
def save_data_to_disk(data, filename='camera_data.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print("Data saved to disk.")

def get_can_bus_info(nusc, nusc_can_bus, sample, scene_name):
    # scene_name = nusc.get('scene', sample['scene_token'])['name']
    sample_timestamp = sample['timestamp']
    try:
        pose_list = nusc_can_bus.get_messages(scene_name, 'pose')
    except:
        return np.zeros(18)  # server scenes do not have can bus information.
    can_bus = []
    # during each scene, the first timestamp of can_bus may be large than the first sample's timestamp
    last_pose = pose_list[0]
    for i, pose in enumerate(pose_list):
        if pose['utime'] > sample_timestamp:
            break
        last_pose = pose
    _ = last_pose.pop('utime')  # useless
    pos = last_pose.pop('pos')
    rotation = last_pose.pop('orientation')
    can_bus.extend(pos)
    can_bus.extend(rotation)
    for key in last_pose.keys():
        print(key, pose[key])
        can_bus.extend(pose[key])  # 16 elements
    can_bus.extend([0., 0.])
    return np.array(can_bus)


def all_camera_dict():
    # Iterate over each scene
    for scene_index, scene in enumerate(nusc.scene):
        print(f"Processing scene {scene_index + 1}/{len(nusc.scene)}")
        sample_token = scene['first_sample_token']
        while sample_token:
            sample = nusc.get('sample', sample_token)
            # print(sample)
            scene_name = nusc.get('scene', sample['scene_token'])['name']

            lidar_token = sample['data']['LIDAR_TOP']
            sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            lidar_cs_record = nusc.get('calibrated_sensor',
                                sd_rec['calibrated_sensor_token'])
            # print(lidar_cs_record)
            # Iterate over each camera for the current sample
            for cam in cams:
                token = sample['data'][cam]
                # Traverse through linked sample data tokens for the current camera
                while token != '':
                    print(token)
                    data = nusc.get('sample_data', token)
                    print(data)
                    can_bus = get_can_bus_info(nusc, nusc_can_bus, data, scene_name)
                    # print(can_bus)
                    # print(data['sample_token'])
                    # print(nusc.get('sample', data['sample_token']))
                    cs_record = nusc.get('calibrated_sensor',
                         data['calibrated_sensor_token'])
                    pose_record = nusc.get('ego_pose', data['ego_pose_token'])
                    # print(cs_record)
                    # print(pose_record)
                    cam_path, box_list, cam_intrinsic = nusc.get_sample_data(token)
                    
                    # print(cam_path, data["is_key_frame"])
                    # print(box_list[0].center, box_list[0].wlh)
                    # print(len(box_list))
                    # Create a dictionary for this specific image
                    image_info = {
                        "cam_name": cam,
                        "is_key_frame": data["is_key_frame"],
                        "timestamp": data["timestamp"],
                        "filename": data['filename']
                    }
                    # Append the image info to the list associated with the camera
                    all_camera_data[cam].append(image_info)

                    # Move to the next image in the sequence
                    token = data["next"]

            # Move to the next sample in the scene
            sample_token = sample['next']

        # Optionally, save data to disk after each scene to manage memory
        # if scene_index % 10 == 0:  # Save after every 10 scenes
        #     save_data_to_disk(all_camera_data)
        #     all_camera_data = {cam: [] for cam in cams}  # Reset the dictionary

    # Save any remaining data at the end of processing
    # save_data_to_disk(all_camera_data)

    print("All data processed and saved.")



# Function to load data from disk
def load_data_from_disk(filename='camera_data.pkl'):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def test_dict():
    # Load the data
    camera_data = load_data_from_disk()

    # Optionally, print some basic info to validate the structure
    print("Loaded data types and counts:")
    for cam, images in camera_data.items():
        print(f"{cam}: {len(images)} images")

    # Check a few entries from each camera
    for cam, images in camera_data.items():
        print(f"\n{cam} sample entries:")
        for image_info in images[:2]:  # Show first 3 entries for each camera
            print(image_info)

    # Validate timestamps and key frames
    for cam, images in camera_data.items():
        timestamps = [img['timestamp'] for img in images]
        key_frames = [img for img in images if img['is_key_frame']]
        print(f"\n{cam} validation:")
        print(f"  Timestamps ordered: {all(timestamps[i] <= timestamps[i + 1] for i in range(len(timestamps) - 1))}")
        print(f"  Key frames: {len(key_frames)} of {len(images)}")

all_camera_dict()
# test_dict()