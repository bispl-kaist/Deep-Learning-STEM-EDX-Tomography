import pathlib
import random
from math import ceil
from glob import glob

import h5py
from scipy.io import loadmat
from torch.utils.data import Dataset
from data.data_transforms import to_tensor
import numpy as np


class SAITData(Dataset):

    def __init__(self, root, transform):

        self.transform = transform
        self.root = root

        input_files = list(root.glob('Input/*.mat'))
        label_files = list(root.glob('Label/*.mat'))
        if not input_files or not label_files:  # If the list is empty for any reason
            raise FileNotFoundError('Sorry! No label files present in this directory. '
                                    'Please check if your disk has been loaded.')

        print(f'Initializing')

        self.input_files = list()
        self.label_files = list()

        for input_file_name in sorted(input_files):
            input_file_name_str = str(input_file_name)
            self.input_files.append(input_file_name_str)

        for label_file_name in sorted(label_files):
            label_file_name_str = str(label_file_name)
            self.label_files.append(label_file_name_str)

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_file_path = self.input_files[idx]
        input_slice = loadmat(input_file_path)
        input_slice = input_slice['x_3chan']
        input_slice = to_tensor(input_slice)

        label_file_path = self.label_files[idx]
        label_slice = loadmat(label_file_path)
        label_slice = label_slice['x_3chan']
        label_slice = to_tensor(label_slice)

        return input_slice, label_slice, input_file_path, label_file_path


class SAITData_inference(Dataset):

    def __init__(self, root, transform):

        self.transform = transform
        self.root = root

        input_files = list(root.glob('Input/*.mat'))
        # input_files = list(root.glob('Input/*/*.mat'))
        if not input_files:  # If the list is empty for any reason
            raise FileNotFoundError('Sorry! No Input files present in this directory. '
                                    'Please check if your disk has been loaded.')

        print(f'Initializing')

        self.input_files = list()

        for input_file_name in sorted(input_files):
            input_file_name_str = str(input_file_name)
            self.input_files.append(input_file_name_str)

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_file_path = self.input_files[idx]
        input_slice = loadmat(input_file_path)
        input_slice = input_slice['x_3chan']
        input_slice = to_tensor(input_slice)

        return input_slice, input_file_path