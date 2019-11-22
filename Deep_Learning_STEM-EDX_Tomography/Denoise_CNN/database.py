import scipy
import scipy.io as spio
import os
from glob import glob
import numpy as np

class Database:
    def __init__(self, param):
        self.input_dir = param.input_dir

        self.train_label_lst = sorted(glob(os.path.join(self.input_dir, 'train', 'label_*.mat')))
        self.train_input_lst = sorted(glob(os.path.join(self.input_dir, 'train', 'input_*.mat')))

        self.val_label_lst = sorted(glob(os.path.join(self.input_dir, 'val', 'label_*.mat')))
        self.val_input_lst = sorted(glob(os.path.join(self.input_dir, 'val', 'input_*.mat')))

        self.test_label_lst = sorted(glob(os.path.join(self.input_dir, 'test', 'label_*.mat')))
        self.test_input_lst = sorted(glob(os.path.join(self.input_dir, 'test', 'input_*.mat')))

        self.train_num  = len(self.train_input_lst)
        self.val_num    = len(self.val_input_lst)
        self.test_num   = len(self.test_input_lst)

        self.__train    = np.arange(0, self.train_num)
        self.__val      = np.arange(0, self.val_num)
        self.__test     = np.arange(0, self.test_num)

        self.kernel_y_size = param.kernel_y_size
        self.kernel_x_size = param.kernel_x_size
        self.kernel_ch_size = param.kernel_ch_size

        self.input_weight = param.input_weight

        self.input_y_size = param.input_y_size
        self.input_x_size = param.input_x_size
        self.input_ch_size = param.input_ch_size

        self.output_y_size = param.output_y_size
        self.output_x_size = param.output_x_size
        self.output_ch_size = param.output_ch_size

        self.load_y_size = param.load_y_size
        self.load_x_size = param.load_x_size
        self.load_ch_size = param.load_ch_size

        if param.patch_y_size == -1:
            self.patch_y_size = param.input_y_size
        else:
            self.patch_y_size = param.patch_y_size

        if param.patch_x_size == -1:
            self.patch_x_size = param.input_x_size
        else:
            self.patch_x_size = param.patch_x_size

        if param.patch_ch_size == -1:
            self.patch_ch_size = param.input_ch_size
        else:
            self.patch_ch_size = param.patch_ch_size

    def get_index(self, shuffle=True, type='train'):
        if type == 'train':
            __idx       = self.__train
            __idx_num   = self.train_num
        elif type == 'val':
            __idx       = self.__val
            __idx_num   = self.val_num
        elif type == 'test':
            __idx       = self.__test
            __idx_num   = self.test_num

        idx = __idx

        if shuffle:
            np.random.shuffle(idx)

        return idx

    def get_data(self, idx, mode='train', type='input'):
        if mode == 'train':
            if type == 'label':
                image = spio.loadmat(self.train_label_lst[idx])['label']
            else:
                image = spio.loadmat(self.train_input_lst[idx])['input']
        elif mode == 'val':
            if type == 'label':
                image = spio.loadmat(self.val_label_lst[idx])['label']
            else:
                image = spio.loadmat(self.val_input_lst[idx])['input']
        elif mode == 'test':
            if type == 'label':
                image = spio.loadmat(self.test_label_lst[idx])['label']
            else:
                image = spio.loadmat(self.test_input_lst[idx])['input']

        return image

    def get_image2patch(self, image):
        patch, nsmp = self._image2patch(image)
        return patch, nsmp

    def get_patch2image(self, patch, type='count'):
        image = self._patch2image(patch, type)
        return image

    def get_database(self, idx, mode='train'):
        label = np.zeros([len(idx), self.patch_y_size, self.patch_x_size, self.output_ch_size])
        input = np.zeros([len(idx), self.patch_y_size, self.patch_x_size, self.patch_ch_size])

        for i, idx_ in enumerate(idx):

            if mode == 'train':
                iy = int(np.floor(np.random.rand(1) * (self.load_y_size - self.patch_y_size)))
                ix = int(np.floor(np.random.rand(1) * (self.load_x_size - self.patch_x_size)))
                label_ = spio.loadmat(self.train_label_lst[idx_])['label']
                input_ = spio.loadmat(self.train_input_lst[idx_])['input']

                if self.input_ch_size == 3:
                    input_ = scipy.ndimage.zoom(input_, (self.load_y_size/self.input_y_size, self.load_x_size/self.input_x_size, 1), order=1)
                else:
                    input_ = scipy.ndimage.zoom(input_, (self.load_y_size / self.input_y_size, self.load_x_size / self.input_x_size), order=1)

                if self.output_ch_size == 3:
                    label_ = scipy.ndimage.zoom(label_, (self.load_y_size/self.input_y_size, self.load_x_size/self.input_x_size, 1), order=1)
                else:
                    label_ = scipy.ndimage.zoom(label_, (self.load_y_size / self.input_y_size, self.load_x_size / self.input_x_size), order=1)

            elif mode == 'val':
                iy = int(np.floor(np.random.rand(1) * (self.input_y_size - self.patch_y_size)))
                ix = int(np.floor(np.random.rand(1) * (self.input_x_size - self.patch_x_size)))
                label_ = spio.loadmat(self.val_label_lst[idx_])['label']
                input_ = spio.loadmat(self.val_input_lst[idx_])['input']

            if self.input_ch_size == 1:
                input_ = np.expand_dims(input_, axis=-1)

            if self.output_ch_size == 1:
                label_ = np.expand_dims(label_, axis=-1)

            __label = label_[iy:iy + self.patch_y_size, ix:ix + self.patch_x_size, :]
            __input = input_[iy:iy + self.patch_y_size, ix:ix + self.patch_x_size, :]

            if (np.random.rand(1) > 0.5):
                __label = np.flip(__label, 0)
                __input = np.flip(__input, 0)

            if (np.random.rand(1) > 0.5):
                __label = np.flip(__label, 1)
                __input = np.flip(__input, 1)

            label[i, :, :, :] = __label
            input[i, :, :, :] = __input

        return label, input

    def add_noise(self, data, type='normal', sgm=0):
        if type == 'normal':
            data = data + sgm*np.random.normal(size=data.shape)
        elif type == 'uniform':
            data = data + sgm*np.random.uniform(size=data.shape)

        return data

    def _image2patch(self, src):
        if np.size(src.shape) == 2:
            src = np.expand_dims(src, -1)

        nimg = [self.input_y_size, self.input_x_size, self.input_ch_size]
        npatch = [self.patch_y_size, self.patch_x_size, self.patch_ch_size]
        nker = [self.kernel_y_size, self.kernel_x_size, self.kernel_ch_size]

        nimg_zp = np.zeros(3, 'int32')
        ncrop = np.zeros(3, 'int32')
        nset = np.zeros(3, 'int32')
        nsmp = 1

        for i in range(0, 3):
            nimg_zp[i] = nimg[i] + 2 * nker[i]
            ncrop[i] = npatch[i] - 2 * nker[i]
            nset[i] = np.ceil(nimg_zp[i] / ncrop[i]).astype('int32')
            nsmp *= nset[i]

        iy_set = (np.linspace(0, nimg_zp[0] - npatch[0], nset[0])).astype('int32')
        ix_set = (np.linspace(0, nimg_zp[1] - npatch[1], nset[1])).astype('int32')
        iz_set = (np.linspace(0, nimg_zp[2] - npatch[2], nset[2])).astype('int32')

        patchy = np.arange(0, npatch[0])
        patchx = np.arange(0, npatch[1])
        patchz = np.arange(0, npatch[2])

        src = np.pad(src, ((nker[0], nker[0]), (nker[1], nker[1]), (nker[2], nker[2])), 'reflect')
        dst = np.zeros((nsmp, npatch[0], npatch[1], npatch[2]), 'float32')

        for iz in range(0, nset[2]):
            for ix in range(0, nset[1]):
                for iy in range(0, nset[0]):
                    pos = nset[0]*nset[1]*iz + nset[0]*ix + iy

                    iy_ = (iy_set[iy] + patchy)[:, np.newaxis, np.newaxis]
                    ix_ = (ix_set[ix] + patchx)[:, np.newaxis]
                    iz_ = (iz_set[iz] + patchz)

                    dst[pos, :, :, :] = src[iy_, ix_, iz_]

        return dst, nsmp

    def _patch2image(self, src, type='count'):
        nimg = [self.input_y_size, self.input_x_size, self.output_ch_size]
        npatch = [self.patch_y_size, self.patch_x_size, self.output_ch_size]
        nker = [self.kernel_y_size, self.kernel_x_size, self.kernel_ch_size]

        nimg_zp = np.zeros(3, 'int32')
        ncrop = np.zeros(3, 'int32')
        nset = np.zeros(3, 'int32')
        nsmp = 1

        for i in range(0, 3):
            nimg_zp[i] = nimg[i] + 2 * nker[i]
            ncrop[i] = npatch[i] - 2 * nker[i]
            nset[i] = np.ceil(nimg_zp[i] / ncrop[i]).astype('int32')
            nsmp *= nset[i]

        iy_set = (np.linspace(0, nimg_zp[0] - npatch[0], nset[0])).astype('int32')
        ix_set = (np.linspace(0, nimg_zp[1] - npatch[1], nset[1])).astype('int32')
        iz_set = (np.linspace(0, nimg_zp[2] - npatch[2], nset[2])).astype('int32')

        patchy = (np.arange(0, npatch[0]))
        patchx = (np.arange(0, npatch[1]))
        patchz = (np.arange(0, npatch[2]))

        cropy = (np.arange(0, ncrop[0]) + nker[0])
        cropx = (np.arange(0, ncrop[1]) + nker[1])
        cropz = (np.arange(0, ncrop[2]) + nker[2])

        cropy_ = cropy[:, np.newaxis, np.newaxis]
        cropx_ = cropx[:, np.newaxis]
        cropz_ = cropz

        bndy_ = (np.arange(nker[0], nimg_zp[0] - nker[0]))[:, np.newaxis, np.newaxis]
        bndx_ = (np.arange(nker[1], nimg_zp[1] - nker[1]))[:, np.newaxis]
        bndz_ = (np.arange(nker[2], nimg_zp[2] - nker[2]))

        if nker[0] == 0:
            wgt_smy = np.ones((npatch[0] - 2 * nker[0], ncrop[1], ncrop[2]), 'float32')
        else:
            nsmy = ncrop[0] - iy_set[1]
            t = np.linspace(np.pi, 2 * np.pi, nsmy)
            wgt_smy = np.ones((npatch[0] - 2 * nker[0]), 'float32')
            wgt_smy[0:nsmy] = (np.cos(t) + 1.0) / 2.0
            wgt_smy = wgt_smy[:, np.newaxis, np.newaxis]
            wgt_smy = np.tile(wgt_smy, (1, ncrop[1], ncrop[2]))

        if nker[1] == 0:
            wgt_smx = np.ones((ncrop[0], npatch[1] - 2 * nker[1], ncrop[2]), 'float32')
        else:
            nsmx = ncrop[1] - ix_set[1]
            t = np.linspace(np.pi, 2 * np.pi, nsmx)
            wgt_smx = np.ones((npatch[1] - 2 * nker[1]), 'float32')
            wgt_smx[0:nsmx] = (np.cos(t) + 1.0) / 2.0
            wgt_smx = wgt_smx[:, np.newaxis]
            wgt_smx = np.tile(wgt_smx, (ncrop[0], 1, ncrop[2]))

        if nker[2] == 0:
            wgt_smz = np.ones((ncrop[0], ncrop[1], npatch[2] - 2 * nker[2]), 'float32')
        else:
            nsmz = ncrop[2] - iz_set[1]
            t = np.linspace(np.pi, 2 * np.pi, nsmz)
            wgt_smz = np.ones((npatch[2] - 2 * nker[2]), 'float32')
            wgt_smz[0:nsmz] = (np.cos(t) + 1.0) / 2.0
            wgt_smz = np.tile(wgt_smz, (ncrop[0], ncrop[1], 1))

        dst = np.zeros([nimg_zp[0], nimg_zp[1], nimg_zp[2]], 'float32')
        wgt = np.zeros([nimg_zp[0], nimg_zp[1], nimg_zp[2]], 'float32')

        for iz in range(0, nset[2]):
            for ix in range(0, nset[1]):
                for iy in range(0, nset[0]):
                    pos = nset[0]*nset[1]*iz + nset[0]*ix + iy

                    iy_ = (iy_set[iy] + cropy[:])[:, np.newaxis, np.newaxis]
                    ix_ = (ix_set[ix] + cropx[:])[:, np.newaxis]
                    iz_ = (iz_set[iz] + cropz[:])

                    if type == 'count':
                        wgt_ = 1
                    else:
                        if iy == 0:
                            wgt_ = np.flip(wgt_smy, 0)
                        elif iy == nset[0] - 1:
                            wgt_ = wgt_smy
                        else:
                            wgt_ = np.flip(wgt_smy, 0) * wgt_smy

                        if ix == 0:
                            wgt_ = wgt_ * np.flip(wgt_smx, 1)
                        elif ix == nset[1] - 1:
                            wgt_ = wgt_ * wgt_smx
                        else:
                            wgt_ = wgt_ * np.flip(wgt_smx, 1) * wgt_smx

                        if iz == 0:
                            wgt_ = wgt_ * np.flip(wgt_smz, 2)
                        elif iz == nset[2] - 1:
                            wgt_ = wgt_ * wgt_smz
                        else:
                            wgt_ = wgt_ * np.flip(wgt_smz, 2) * wgt_smz

                    src_ = src[pos, :, :, :]
                    dst[iy_, ix_, iz_] = dst[iy_, ix_, iz_] + src_[cropy_, cropx_, cropz_] * wgt_
                    wgt[iy_, ix_, iz_] = wgt[iy_, ix_, iz_] + wgt_

        dst = (dst[bndy_, bndx_, bndz_]).astype('float32')
        wgt = (wgt[bndy_, bndx_, bndz_]).astype('float32')

        if type == 'count':
            dst = dst/wgt

        return dst, wgt
