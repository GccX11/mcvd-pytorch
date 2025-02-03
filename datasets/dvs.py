# https://github.com/edenton/svg/blob/master/data/kth.py
import numpy as np
import os
import pickle
import torch
import tqdm

from natsort import natsorted
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# TODO: we probably want to sample a single short sequence from each example
#       rather than random subsequences like we are doing...
#       but, I need to read the paper to figure this out

class DVSDataset(Dataset):
    def __init__(self, data_dir, frames_per_sample=15, train=True, random_time=True, random_horizontal_flip=False,
                 total_videos=-1, with_target=True, start_at=0, max_exs=5):

        self.data_dir = os.path.join(data_dir, 'train') if train else os.path.join(data_dir, 'test')
        self.train = train
        self.frames_per_sample = frames_per_sample
        self.random_time = random_time
        self.random_horizontal_flip = random_horizontal_flip
        self.total_videos = total_videos            # If we wish to restrict total number of videos (e.g. for val)
        self.with_target = with_target
        self.start_at = start_at

        # load the frames from 6 classes (just for simplicity)
        # break into sequences of length=frames_per_sample and put in list

        # Get all class folders
        class_folders = sorted(os.listdir(self.data_dir))
        class_to_idx = {cls: i for i, cls in enumerate(class_folders)}

        # Process each class folder
        self.sequences = []
        for class_folder in class_folders:
            class_path = os.path.join(self.data_dir, class_folder)
            if not os.path.isdir(class_path):
                continue
            
            if class_folder.startswith('.'):
                continue

            print('Loading', os.path.basename(class_path))
            
            # Get all sequence subfolders in this class
            sequence_folders = []
            for f in os.listdir(class_path):
                # skip non-example folders
                if not f.startswith('ex_'):
                    continue
                ex_num = int(f.split('_')[1])
                # only load up to max_exs examples
                if max_exs is not None and ex_num >= max_exs:
                    continue
                sequence_folders.append(f)
            sequence_folders = natsorted(sequence_folders)

            for sequence_folder in tqdm.tqdm(sequence_folders):
                sequence_path = os.path.join(class_path, sequence_folder)
                if not os.path.isdir(sequence_path):
                    continue
                    
                # Get all image files in this sequence
                images = natsorted(os.listdir(sequence_path))
                total_frames = len(images)
                
                # Split into sequences of frames_per_sample
                for i in range(0, total_frames - frames_per_sample + 1):
                    sequence_images = []
                    for j in range(frames_per_sample):
                        img_path = os.path.join(sequence_path, images[i + j])
                        with Image.open(img_path) as img:
                            sequence_images.append(torch.from_numpy(np.array(img, dtype=np.int8)))
                        
                    if with_target:
                        self.sequences.append((sequence_images, class_to_idx[class_folder]))
                    else:
                        self.sequences.append(sequence_images)

        print(f"Dataset length: {self.__len__()}")


    def __len__(self):
        return self.total_videos if self.total_videos > 0 else len(self.sequences)


    def __getitem__(self, index):
        # Get the sequence (images and target if with_target is True)
        if self.with_target:
            sequence_images, target = self.sequences[index]
        else:
            sequence_images = self.sequences[index] 
        
        # Convert images to tensors
        # TODO: figure out what the horizontal flip does in the orignal dataset
        #       maybe it just generates extra flipped sequences?
        # prefinals = []
        # flip_p = np.random.randint(2) == 0 if self.random_horizontal_flip else 0
        # for img in sequence_images:
        #     arr = transforms.RandomHorizontalFlip(flip_p)(transforms.ToTensor()(img))
        #     prefinals.append(arr)
            
        if self.with_target:
            return torch.stack(sequence_images).unsqueeze(1), torch.tensor(target)
        else:
            return torch.stack(sequence_images).unsqueeze(1)
