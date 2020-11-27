import torch
import numpy as np
import torch.nn.functional as F
import random
from tvtk.api import tvtk, write_data

def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.
    Args:
        data (np.array): Input numpy array
    Returns:
        torch.Tensor: PyTorch version of data
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)
    data = data.astype(np.float32)
    return torch.from_numpy(data)


def normalize_im(tensor):
    large = torch.max(tensor)
    small = torch.min(tensor)
    diff = large - small

    normalized_tensor = (tensor.clamp(min=small, max=large) - small) * (torch.tensor(1) / diff)

    return normalized_tensor


def extract_patch(inputs, targets, patch_size=128):
    assert inputs.dim() == 4, "Tensor should be batched and have dimension 4"
    assert isinstance(inputs, torch.Tensor), "Input data should be torch tensor"
    assert isinstance(targets, torch.Tensor), "Input data should be torch tensor"

    with torch.no_grad():
        h = inputs.shape[-2]
        w = inputs.shape[-1]

        start_h = random.randint(0, h - (patch_size + 1))
        start_w = random.randint(0, w - (patch_size + 1))

        patch_inputs = inputs[:, :, start_h:start_h + patch_size, start_w:start_w + patch_size]
        patch_targets = targets[:, start_h:start_h + patch_size, start_w:start_w + patch_size]

    assert patch_inputs.shape[-1] == patch_size
    assert patch_inputs.shape[-1] == patch_size

    return patch_inputs, patch_targets


def extract_patch_transform_single(inputs, patch_size=128):
    assert isinstance(inputs, torch.Tensor), "Input data should be torch tensor"

    with torch.no_grad():
        h = inputs.shape[-2]
        w = inputs.shape[-1]

        start_h = random.randint(0, h - (patch_size + 1))
        start_w = random.randint(0, w - (patch_size + 1))

        patch_inputs = inputs[:, :, start_h:start_h + patch_size, start_w:start_w + patch_size]

    return patch_inputs


# Use this function when you want to extract patch inside the input_transform function
def extract_patch_transform(inputs, targets, patch_size=128):
    assert isinstance(inputs, torch.Tensor), "Input data should be torch tensor"
    assert isinstance(targets, torch.Tensor), "Input data should be torch tensor"

    with torch.no_grad():
        h = inputs.shape[-2]
        w = inputs.shape[-1]

        start_h = random.randint(0, h - (patch_size + 1))
        start_w = random.randint(0, w - (patch_size + 1))

        patch_inputs = inputs[:, :, start_h:start_h + patch_size, start_w:start_w + patch_size]
        patch_targets = targets[:, :, start_h:start_h + patch_size, start_w:start_w + patch_size]

    return patch_inputs, patch_targets


def extract_patch_transform_vol(inputs, targets, patch_size=(128, 128, 128)):
    assert isinstance(inputs, torch.Tensor), "Input data should be torch tensor"
    assert isinstance(targets, torch.Tensor), "Input data should be torch tensor"

    with torch.no_grad():
        h = inputs.shape[-3]
        w = inputs.shape[-2]
        d = inputs.shape[-1]
        if patch_size[0] == 256:
            start_h = 0
        else:
            start_h = random.randint(0, h - (patch_size[0] + 1))
        if patch_size[1] == 256:
            start_w = 0
        else:
            start_w = random.randint(0, w - (patch_size[1] + 1))
        if patch_size[2] == 256:
            start_d = 0
        else:
            start_d = random.randint(0, d - (patch_size[2] + 1))

        patch_inputs = inputs[:, :, start_h:start_h + patch_size[0], start_w:start_w + patch_size[1], start_d:start_d + patch_size[2]]
        patch_targets = targets[:, :, start_h:start_h + patch_size[0], start_w:start_w + patch_size[1], start_d:start_d + patch_size[2]]

    return patch_inputs, patch_targets


def extract_patch_transform_vol_mask(inputs, targets, mask, patch_size=64):
    assert isinstance(inputs, torch.Tensor), "Input data should be torch tensor"
    assert isinstance(targets, torch.Tensor), "Input data should be torch tensor"

    with torch.no_grad():
        h = inputs.shape[-3]
        w = inputs.shape[-2]
        d = inputs.shape[-1]
        start_h = random.randint(0, h - (patch_size + 1))
        start_w = random.randint(0, w - (patch_size + 1))
        start_d = random.randint(0, d - (patch_size + 1))

        patch_inputs = inputs[:, :, start_h:start_h + patch_size, start_w:start_w + patch_size, start_d:start_d + patch_size]
        patch_targets = targets[:, :, start_h:start_h + patch_size, start_w:start_w + patch_size, start_d:start_d + patch_size]
        patch_mask = mask[:, :, start_h:start_h + patch_size, start_w:start_w + patch_size,
                        start_d:start_d + patch_size]

    return patch_inputs, patch_targets, patch_mask


def extract_patch_transform_inference(inputs, patch_size=128, stride=64):
    '''
    patch_size: The size of patch that was trained with
    stride:
    '''
    assert isinstance(inputs, torch.Tensor), "Input data should be torch tensor"

    with torch.no_grad():
        ps = patch_size
        s = stride
        # Actual size of total volume
        h = inputs.shape[-3]
        w = inputs.shape[-2]
        d = inputs.shape[-1]

        hs = (h // stride) - 1
        ws = (w // stride) - 1
        ds = (d // stride) - 1

        inputs_list = list()
        for i in range(hs):
            for j in range(ws):
                for k in range(ds):
                    inputs_list.append(inputs[:, :, i*s:i*s+ps, j*s:j*s+ps, k*s:k*s+ps])

    return inputs_list


def extract_patch_transform_inference_general(inputs, patch_size=(256, 256, 32), stride=(0, 0, 16)):
    '''
    general version of the function "extract_patch_transform_inference"
    receives a tuple of patch_size. (e.g. patch_size = (256, 256, 32))
    receives a tuple of stride. (e.g. stride = (0, 0, 16))
    '''
    assert isinstance(inputs, torch.Tensor), "Input data should be torch tensor"

    with torch.no_grad():
        ps_h, ps_w, ps_d = patch_size
        s_h, s_w, s_d = stride
        # Actual size of total volume
        h = inputs.shape[-3]
        w = inputs.shape[-2]
        d = inputs.shape[-1]

        if s_h == 0:
            hs = 1
        else:
            hs = ((h - ps_h) // s_h) + 1
        if s_w == 0:
            ws = 1
        else:
            ws = ((w - ps_w) // s_w) + 1
        if s_d == 0:
            ds = 1
        else:
            ds = ((d - ps_d) // s_d) + 1

        inputs_list = list()
        for i in range(hs):
            for j in range(ws):
                for k in range(ds):
                    inputs_list.append(inputs[:, :, i*s_h:i*s_h+ps_h, j*s_w:j*s_w+ps_w, k*s_d:k*s_d+ps_d])

    return inputs_list


def extract_patch_transform_inference_gather(outputs_stack, full_vol, patch_size=128, stride=64):
    '''
    full_size : Empty tensor that has the size of full volume e.g. 3 x 256 x 256 x 256
    patch_size: The size of patch that was trained with
    stride:
    '''
    assert isinstance(outputs_stack, list), "Output stack must be a list of tensors"
    wgt = torch.zeros_like(full_vol)  # Weight map that should be divided so that the scale remains 1 at every point
    ps = patch_size
    s = stride

    b, c, h, w, d = full_vol.shape
    hs = (h // stride) - 1
    ws = (w // stride) - 1
    ds = (d // stride) - 1

    flg = 0
    for i in range(hs):
        for j in range(ws):
            for k in range(ds):
                full_vol[:, :, i*s:i*s+ps, j*s:j*s+ps, k*s:k*s+ps] += outputs_stack[flg]
                wgt[:, :, i*s:i*s+ps, j*s:j*s+ps, k*s:k*s+ps] += 1
                flg += 1

    full_vol /= wgt

    return full_vol.squeeze()


def extract_patch_transform_inference_gather_general(outputs_stack, full_vol, patch_size=(128, 128, 128), stride=(64, 64, 64)):
    '''
    General version of extract_patch_transform_inference_gather
    full_vol : Empty tensor that has the size of full volume e.g. 3 x 256 x 256 x 256
    patch_size: tuple of patch size e.g. (128, 128, 128)
    stride: tuple of stride e.g. (64, 64, 64)
    '''
    assert isinstance(outputs_stack, list), "Output stack must be a list of tensors"
    wgt = torch.zeros_like(full_vol)  # Weight map that should be divided so that the scale remains 1 at every point
    ps_h, ps_w, ps_d = patch_size
    s_h, s_w, s_d = stride

    b, c, h, w, d = full_vol.shape
    if s_h == 0:
        hs = 1
    else:
        hs = ((h - ps_h) // s_h) + 1
    if s_w == 0:
        ws = 1
    else:
        ws = ((w - ps_w) // s_w) + 1
    if s_d == 0:
        ds = 1
    else:
        ds = ((d - ps_d) // s_d) + 1

    flg = 0
    for i in range(hs):
        for j in range(ws):
            for k in range(ds):
                full_vol[:, :, i*s_h:i*s_h+ps_h, j*s_w:j*s_w+ps_w, k*s_d:k*s_d+ps_d] += outputs_stack[flg]
                wgt[:, :, i*s_h:i*s_h+ps_h, j*s_w:j*s_w+ps_w, k*s_d:k*s_d+ps_d] += 1
                flg += 1

    full_vol /= wgt

    return full_vol.squeeze()