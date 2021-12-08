from __future__ import print_function, division
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, transform
from skimage.transform import rescale, resize
import os
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms, utils
import random
from skimage.color import rgba2rgb
import math

class EventDataset(Dataset):
    def __init__(self, args,csv_file, root_dir, transform=None, slice_frames=None,select_ratio=1.0, select_range=None):
        assert select_ratio >= -1.0 and select_ratio <= 1.0

        camera_csv = pd.read_csv(csv_file)
        csv_len = len(camera_csv)
        if slice_frames:
            csv_selected = camera_csv[0:0]  # empty dataframe
            for start_idx in range(0, csv_len, slice_frames):
                if select_ratio > 0:
                    end_idx = int(start_idx + slice_frames * select_ratio)
                else:
                    start_idx, end_idx = int(
                        start_idx + slice_frames * (1 + select_ratio)), start_idx + slice_frames

                if end_idx > csv_len:
                    end_idx = csv_len
                if start_idx > csv_len:
                    start_idx = csv_len
                csv_selected = csv_selected.append(
                    camera_csv[start_idx:end_idx])
            self.camera_csv = csv_selected
        elif select_range:
            csv_selected = camera_csv.iloc[select_range[0]: select_range[1]]
            self.camera_csv = csv_selected
        else:
            self.camera_csv = camera_csv

        self.root_dir = root_dir
        self.transform = transform
        self.mean = {}
        self.std = {}
        for key in ['steering']:
            self.mean[key] = np.mean(camera_csv[key])
            self.std[key] = np.std(camera_csv[key])
        
    def __len__(self):
        return len(self.camera_csv)
    
    def read_data_single(self, idx):
        path_dvs = self.camera_csv['event_frame'].iloc[idx]
        path_aps = self.camera_csv['rgb_data'].iloc[idx]
        dvs_image = io.imread(path_dvs)
        aps_image = io.imread(path_aps)
        aps_image=aps_image/255
        angle = self.camera_csv['steering'].iloc[idx]

        if self.args.data_name == 'ddd':
            dvs_image = rgba2rgb(dvs_image)
            aps_image = rgba2rgb(aps_image)
            angle = math.radians(angle)
        elif self.args.data_name == 'drfuser':
            aps_image=np.resize(aps_image,(260,346,3))
            angle = math.radians(angle)
      

        # speed = self.camera_csv['speed'].iloc[idx]

        if self.transform:
            dvs_image_transformed = self.transform(dvs_image)
            aps_image_transformed = self.transform(aps_image)
            del dvs_image
            del aps_image
            dvs_image = dvs_image_transformed
            aps_image = aps_image_transformed
        angle_t = torch.tensor(angle,dtype=torch.float32)
        # speed_t = torch.tensor(speed)
        del angle #speed

        return dvs_image,aps_image,angle_t,path_dvs,path_aps #,speed_t

    def read_data(self, idx):
        if isinstance(idx, list):
            data = None
            for i in idx:
                new_data = self.read_data(i)
                if data is None:
                    data = [[] for _ in range(len(new_data))]
                for i, d in enumerate(new_data):
                    data[i].append(new_data[i])
                del new_data

            # we don't stack timestamp and frame_id since those are string data
            for stack_idx in [0, 1, 2, 3, 4]:
                data[stack_idx] = torch.stack(data[stack_idx])

            return data

        else:
            return self.read_data_single(idx)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.read_data(idx)

        sample = {'dvs_image': data[0],
                  'aps_image': data[1],
                  'angle': data[2],
                  'dvs_filename': data[3],
                  'aps_filename':data[4]}
                #   'speed': data[3]}

        del data

        return sample
    
