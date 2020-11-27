import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

from data.data_transforms import nchw_to_kspace, ifft2, fft2, complex_abs, ifft1, fft1, \
    root_sum_of_squares, center_crop, complex_center_crop, extract_patch_transform_inference_gather, \
    extract_patch_transform_inference_gather_general


class SingleOutputTransform(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, extra_params):
        return outputs


class OutputScalePatchTransform_avg(nn.Module):
    def __init__(self, patch_size, stride):
        super().__init__()
        if len(patch_size) != len(stride):
            KeyError(f'Length of patch_size {len(patch_size)} and stride {len(stride)} was given.'
                     f'They should be the same')
        self.patch_size = patch_size
        self.stride = stride

    def forward(self, output_stack, full_vol, extra_params, iter):
        patch_size = self.patch_size[iter]
        stride = self.stride[iter]
        recon_vol = extract_patch_transform_inference_gather_general(output_stack, full_vol,
                                                                     patch_size=patch_size, stride=stride)

        return recon_vol * extra_params['scales']