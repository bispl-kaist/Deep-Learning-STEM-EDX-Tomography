import torch
import torch.nn.functional as F

import numpy as np
import random

from data.data_transforms import to_tensor, ifft2, fft2, complex_abs, apply_info_mask, kspace_to_nchw, ifft1, fft1, \
    complex_center_crop, center_crop, root_sum_of_squares, k_slice_to_chw, ej_kslice_to_chw, ej_permute, ej_permute_bchw, \
    extract_patch_transform, extract_patch_transform_proj, extract_patch_transform_single, extract_patch_transform_vol, \
    extract_patch_transform_inference, extract_patch_transform_vol_mask, extract_patch_transform_inference_general


class Prefetch2Device:
    """
    Fetches input data to GPU device.
    Using this to minimize overhead from passing tensors on device from one process to another.
    Also, on HDD devices this will give the data gathering process more time to get data.

    """
    def __init__(self, device):
        self.device = device

    def __call__(self, input, target, input_file_name, slice_file_name):

        if input.ndim != 3:  # Prevents possible errors.
            raise TypeError('Invalid slice dimensions. Should have dimension of 3 - H x W x C')
        if target.ndim != 3:  # Prevents possible errors.
            raise TypeError('Invalid slice dimensions. Should have dimension of 3 - H x W x C')

        # I hope that async copy works for passing between processes but I am not sure.
        input_slice = to_tensor(input).to(device=self.device)
        target_slice = to_tensor(target).to(device=self.device)

        return input_slice, target_slice, input_file_name, slice_file_name


class Prefetch2DeviceVal:
    """
    Fetches input data to GPU device.
    Using this to minimize overhead from passing tensors on device from one process to another.
    Also, on HDD devices this will give the data gathering process more time to get data.

    """
    def __init__(self, device):
        self.device = device

    def __call__(self, target, slice_file_name):
        if target.ndim != 3:  # Prevents possible errors.
            raise TypeError('Invalid slice dimensions. Should have dimension of 3 - H x W x C')

        # I hope that async copy works for passing between processes but I am not sure.
        target_slice = to_tensor(target).to(device=self.device)

        return target_slice, slice_file_name


class PreProcessScale_random:
    def __init__(self, device, use_seed=True, divisor=1, patch_size=None):
        self.device = device
        self.use_seed = use_seed
        self.divisor = divisor
        self.patch_size = patch_size

    def __call__(self, input_slice, target_slice, input_fname, target_fname):
        assert isinstance(input_slice, torch.Tensor), 'input slice must be torch tensor type'
        assert isinstance(target_slice, torch.Tensor), 'target slice must be torch tensor type'

        with torch.no_grad():
            scale_std = torch.std(input_slice)  # Maybe I should scale per each slice, not per each batch?
            scaling = 1 / scale_std
            input_slice = input_slice * scaling
            target_slice = target_slice * scaling

            input_slice = input_slice.permute(0, 4, 1, 2, 3)
            target_slice = target_slice.permute(0, 4, 1, 2, 3)
            # pick a random patch size from a given list
            patch_size = self.patch_size[random.choice([0, 1, 2, 3])]
            input_slice, target_slice = extract_patch_transform_vol(input_slice, target_slice, patch_size)

            input_slice = input_slice.to(self.device)
            target_slice = target_slice.to(self.device)

            extra_params = {'scales': scale_std}

        return input_slice, target_slice, input_fname, target_fname, extra_params


class PreProcessInfer_avg:
    def __init__(self, device, use_seed=True, divisor=1, patch_size=128, stride=64):
        self.device = device
        self.use_seed = use_seed
        self.divisor = divisor

        if len(patch_size) != len(stride):
            KeyError(f'Length of patch_size {len(patch_size)} and stride {len(stride)} was given.'
                     f'They should be the same')
        self.iter = len(patch_size)
        self.patch_size = patch_size
        self.stride = stride

    def __call__(self, input_slice, input_fname, iter):
        assert isinstance(input_slice, torch.Tensor)

        with torch.no_grad():
            scale_std = torch.std(input_slice)  # Maybe I should scale per each slice, not per each batch?
            scaling = 1 / scale_std
            input_slice = input_slice * scaling
            input_slice = input_slice.permute(0, 4, 1, 2, 3)

            stride = self.stride[iter]
            patch_size = self.patch_size[iter]
            input_stack = extract_patch_transform_inference_general(input_slice, stride=stride, patch_size=patch_size)
            extra_params = {'scales': scale_std}

        return input_stack, input_fname, extra_params