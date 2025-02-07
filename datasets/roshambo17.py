# https://github.com/edenton/svg/blob/master/data/kth.py
import numpy as np
import os
import cv2
import pickle
import tqdm
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# https://ieeexplore.ieee.org/document/8050403
# https://www.research-collection.ethz.ch/handle/20.500.11850/704324
# CUDA_VISIBLE_DEVICES=0 python main.py --config configs/kth64_big.yml --data_path /home/matt/DATA/KTH/h5 --exp kth --ni

# [x] load the AVI files
# [ ] figure out how to index into the individual sequences in the AVI files...
# [-] load the roshambo aedat files
# [ ] try the diffusion model on these representations
# [ ] try RNN on these representations
# [ ] try SSM on these representations


LABEL_MAP = {
    "rock": 0,
    "paper": 1,
    "scissors": 2
}


def inspect_aedat_file(filename):
    with open(filename, 'rb') as f:
        # Read first 1000 bytes to inspect header
        initial_bytes = f.read(1000)
        print("First 100 bytes as hex:")
        print(initial_bytes[:100].hex())
        print("\nFirst 100 bytes as ASCII (if possible):")
        print(initial_bytes[:100])
        print("\nTotal file size:", f.seek(0, 2))
        f.seek(0)
        print("\nFirst few lines:")
        for i in range(5):
            print(f.readline())

def read_aedat2(filename):
    """
    Read AEDAT2 file format
    Returns: numpy array of events with columns (timestamp, x, y, polarity)
    """
    with open(filename, 'rb') as f:
        # Skip header (look for first non-# line)
        while True:
            line = f.readline()
            if not line.startswith(b'#'):
                break
        
        # Go back 1 line since we read the first data byte
        f.seek(-len(line), 1)
        
        # Read the rest as binary data
        data = f.read()
        
        # Convert bytes to array of uint32
        events = np.frombuffer(data, dtype=np.uint32)
        
        # Reshape to (N, 2) array - pairs of (address, timestamp)
        events = events.reshape(-1, 2)
        
        # Extract address and timestamp
        addresses = events[:, 0]
        timestamps = events[:, 1]
        
        # For DVS128 camera:
        # Bit 0: Polarity (0: OFF, 1: ON)
        # Bits 1-7: Y address (0-127)
        # Bits 8-14: X address (0-127)
        polarity = addresses & 1
        y = (addresses >> 1) & 0x7F
        x = (addresses >> 8) & 0x7F
        
        return np.column_stack((timestamps, x, y, polarity))
    

class Roshambo17AVIDataset(Dataset):
    def __init__(self, data_dir, frames_per_sample=20, train=True,
                 start_at=0, skip_frame=1, 
                 random_horizontal_flip=True, with_target=True):
        
        self.frames_per_sample = frames_per_sample
        self.random_horizontal_flip = random_horizontal_flip
        self.with_target = with_target

        if train:
            data_dir = os.path.join(data_dir, "train")
        else:
            data_dir = os.path.join(data_dir, "test")

        self.sequences = []
        # load the AVI files from the data directory
        for filename in tqdm.tqdm(os.listdir(data_dir)):
            if not filename.startswith('background'):
                # decode the file
                cap = cv2.VideoCapture(os.path.join(data_dir, filename))
                num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                #print(f"Found AVI file: {filename} Number of frames: {num_frames}")
                frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
                #print(f"Frame rate: {frame_rate}")
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                #print(f"Frame size: {frame_width}x{frame_height}")

                label = os.path.basename(filename).split('_')[0]

                self.read_frames(cap, label, frames_per_sample=frames_per_sample,
                                start_at=start_at, skip_frame=skip_frame)

    def read_frames(self, cap, label, frames_per_sample=None,
                    start_at=0, skip_frame=1):
        # read frames
        actual_i = 0
        frames =[]
        while True:
            if actual_i < start_at:
                while True:
                    ret, frame = cap.read()
                    actual_i += 1
                    if actual_i % skip_frame == 0:
                        break
                continue
            ret, frame = cap.read()
            if not ret:
                break
            # convert to grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame)

            while True:
                ret, frame = cap.read()
                actual_i += 1
                if actual_i % skip_frame == 0:
                    break
        
        # release the video capture object
        cap.release()
        # add the sequence to the sequences
        self.sequences.append((frames, label))
        
    def read_frames0(self, cap, label, frames_per_sample=10, 
                     start_at=0, skip_frame=1):
        # read frames
        actual_i = 0
        i = 0
        frames = np.zeros((frames_per_sample, 1, 64, 64))
        while True:
            if actual_i < start_at:
                while True:
                    ret, frame = cap.read()
                    actual_i += 1
                    if actual_i % skip_frame == 0:
                        break
                continue
            if i == frames_per_sample:
                break
            ret, frame = cap.read()
            if not ret:
                break
            # convert to grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames[i, 0, :, :] = frame

            i += 1
            while True:
                ret, frame = cap.read()
                actual_i += 1
                if actual_i % skip_frame == 0:
                    break
        
        # release the video capture object
        cap.release()
        # add the sequence to the sequences
        self.sequences.append((frames, label))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        flip_p = np.random.randint(2) == 0 if self.random_horizontal_flip else 0

        frames, label = self.sequences[index]
        if len(frames) > self.frames_per_sample:
            start_idx = np.random.randint(0, len(frames) - self.frames_per_sample)
            selected_frames = frames[start_idx:start_idx + self.frames_per_sample]
        else:
            selected_frames = frames[:self.frames_per_sample]
        
        prefinals = []
        for img in selected_frames:
            # Convert each frame to tensor and apply horizontal flip if needed
            transformed = transforms.RandomHorizontalFlip(flip_p)(transforms.ToTensor()(img))
            prefinals.append(transformed)

        if self.with_target:
            return torch.stack(prefinals), torch.tensor(LABEL_MAP[label])
        else:
            return torch.stack(prefinals)


class Roshambo17DVSDataset(Dataset):
    def __init__(self, data_dir, events_per_sample=5, train=True,
                 start_at=0, with_target=True):
        
        self.events_per_sample = events_per_sample
        self.start_at = start_at
        self.with_target = with_target
        
        # iterate over all aedat files and load the events
        self.events = []
        for filename in tqdm.tqdm(os.listdir(data_dir)):
            if filename.endswith(".aedat") and not filename.startswith('background'):
                try:
                    events = read_aedat2(os.path.join(data_dir, filename))
                    label = os.path.basename(filename).split('_')[0]
                    self.events.append((events, label))
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
        
        print(f"Dataset length: {self.__len__()}")

    def __len__(self):
        return len(self.events)

    def __getitem__(self, index):
        # pair the events with the labels and return a random sample of events
        # with the corresponding label in __getitem__()
        events, label = self.events[index]
        # randomly select a sample of events
        selected_events = events[np.random.choice(events.shape[0], self.events_per_sample)]
        # create a tensor of the events
        prefinals = []
        for event in selected_events:
            # Convert each frame to tensor and apply horizontal flip if needed
            prefinals.append(torch.tensor(event))

        if self.with_target:
            return torch.stack(prefinals), torch.tensor(LABEL_MAP[label])
        else:
            return torch.stack(prefinals)


if __name__ == "__main__":
    dataset = Roshambo17AVIDataset("/home/matt/DATA/ROSHAMBO17/recordings/avi1000")
