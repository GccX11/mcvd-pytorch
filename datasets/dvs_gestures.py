import tonic
import torch
import numpy as np

# Define the sensor size and time window for frame aggregation
sensor_size = (128, 128, 2)
time_window = 20000  # microseconds (100ms)

# Create the transform to convert events to frames
frame_transform = tonic.transforms.ToFrame(sensor_size=sensor_size, time_window=time_window)

# Load the dataset
dataset = tonic.datasets.DVSGesture(save_to='/home/matt/DATA/', 
                                    train=True, 
                                    transform=frame_transform)

# Access a sample
events, label = dataset[0]

# go through each data point in the dataset
# and save it as a video in a folder for each label
import os
import cv2
import tqdm

# Create a folder to save the videos
save_folder = '/home/matt/DATA/DVSGesture/videos_pos_64'

label_map = {0:  'hand_clapping',
             1:  'right_hand_wave',
             2:  'left_hand_wave',
             3:  'right_arm_clockwise',
             4:  'right_arm_counter_clockwise',
             5:  'left_arm_clockwise',
             6:  'left_arm_counter_clockwise',
             7:  'arm_roll',
             8:  'air_drums',
             9:  'air_guitar',
             10: 'other_gestures'}

label_counts = {0: 0,
                1: 0,
                2: 0,
                3: 0,
                4: 0,
                5: 0,
                6: 0,
                7: 0,
                8: 0,
                9: 0,
                10: 0}

for i in range(len(dataset)):
    frames, label = dataset[i]
    label_name = label_map[label]
    #frames = frame_transform(events)
    video_folder = os.path.join(save_folder, str(label_name), f'ex_{label_counts[label]}')
    os.makedirs(video_folder, exist_ok=True)

    # create an AVI file
    #video_path = os.path.join(video_folder, f'video_{i}.avi')
    #out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'MJPG'), 1, (128, 128))
    # accumulate the frames first into a 3D array
    # figure out the max and min values of the frames
    # normalize the frames to [0, 255]
    all_frames = np.zeros((len(frames), 64, 64))
    for j, frame in enumerate(frames):
        # downsample the frames to 64x64
        frame = cv2.resize(frame[0,:,:], (64, 64))
        all_frames[j] = frame
    max_val = np.max(all_frames)
    min_val = np.min(all_frames)
    all_frames = (all_frames - min_val) / (max_val - min_val) * 255
    all_frames = all_frames.astype(np.uint8)
    for j, frame in enumerate(all_frames):
        cv2.imwrite(os.path.join(video_folder, f'frame_{j}.png'), frame)

    label_counts[label] += 1

    print(f'Saved video {i} in folder {label}')
    # if i == 10:
    #     break
