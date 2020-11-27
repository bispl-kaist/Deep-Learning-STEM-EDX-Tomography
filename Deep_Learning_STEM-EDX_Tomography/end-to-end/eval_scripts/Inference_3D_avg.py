import torch
from torch import nn, optim, multiprocessing
from torch.utils.data import DataLoader
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.run_utils import get_logger
from metrics.new_1d_ssim import SSIM
from scipy.io import savemat
import time

from data.data_transforms import fake_input_gen, nchw_to_kspace, np2vtk, normalize_im_np
from scipy.io import loadmat, savemat


class Infer:

    def __init__(self, args, model, eval_loader, eval_input_transform, eval_output_transform):

        multiprocessing.set_start_method(method='spawn')

        self.logger = get_logger(name=__name__, save_file=args.log_path / args.run_name)

        # Checking whether inputs are correct.
        assert isinstance(model, nn.Module), '`model` must be a Pytorch Module.'
        assert isinstance(eval_loader, DataLoader),'`train_loader` and `val_loader` must be Pytorch DataLoader objects.'

        assert callable(eval_input_transform) and callable(eval_output_transform), \
            'input/output_transforms must be callable functions.'

        self.device = args.device
        self.name = args.name
        self.model = model
        self.eval_loader = eval_loader
        self.eval_input_transform = eval_input_transform
        self.eval_output_transform = eval_output_transform

        self.verbose = args.verbose
        self.batch_size = args.batch_size
        self.use_slice_metrics = args.use_slice_metrics
        self.use_residual = args.use_residual
        self.writer = SummaryWriter(str(args.log_path))
        self.ssim = SSIM(filter_size=7).to(device=args.device)  # Needed to cache the kernel.

        self.save_fdir = args.save_fdir
        self.iter = 3

    def inference_patchvol(self, args):
        self.logger.info('Starting inference')
        self.model.eval()
        torch.autograd.set_grad_enabled(False)

        data_loader = enumerate(self.eval_loader, start=1)
        if not self.verbose:
            data_loader = tqdm(data_loader, total=len(self.eval_loader.dataset))

        save_fdir = Path(args.save_fdir)
        save_fdir.mkdir(exist_ok=True)


        for step, data in data_loader:
            tic = time.time()
            for i in range(self.iter):
                print(f'Processing {i+1} / {self.iter}')
                output_stack = list()
                full_vol = torch.zeros(1, 3, 256, 256, 256)
                input_stack, input_fname, extra_params = self.eval_input_transform(*data, i)
                for cnt, inputs in enumerate(input_stack):
                    print(f'Processing input stack {cnt+1}/{len(input_stack)}')
                    inputs = inputs.to(self.device)
                    outputs = self.model(inputs)
                    if self.use_residual:
                        outputs += inputs
                    outputs = outputs.detach().cpu()
                    output_stack.append(outputs)

                if i == 0:
                    recons = self.eval_output_transform(output_stack, full_vol, extra_params, i)
                else:
                    recons += self.eval_output_transform(output_stack, full_vol, extra_params, i)
            recons /= self.iter
            toc = time.time() - tic
            print(f'Elapsed time per specimen: {toc}')
            specimen_num = input_fname[0].split('/')[-1][-6:-4]
            save_fdir_s = Path(self.save_fdir) / specimen_num
            save_fdir_s.mkdir(exist_ok=True)

            se_recons = recons[0, :, :, :].squeeze().numpy()
            s_recons = recons[1, :, :, :].squeeze().numpy()
            zn_recons = recons[2, :, :, :].squeeze().numpy()

            se_recons[se_recons < 0] = 0
            s_recons[s_recons < 0] = 0
            zn_recons[zn_recons < 0] = 0

            se_save_dir_mat = str(save_fdir_s) + '/se.mat'
            s_save_dir_mat = str(save_fdir_s) + '/s.mat'
            zn_save_dir_mat = str(save_fdir_s) + '/zn.mat'

            savemat(se_save_dir_mat, {'se': se_recons})
            savemat(s_save_dir_mat, {'s': s_recons})
            savemat(zn_save_dir_mat, {'zn': zn_recons})

