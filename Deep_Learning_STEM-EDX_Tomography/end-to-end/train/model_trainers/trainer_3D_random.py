import torch
from torch import nn, optim, multiprocessing
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from time import time
from collections import defaultdict

from utils.run_utils import get_logger
from utils.train_utils import CheckpointManager, make_grid_triplet, make_k_grid, make_input_triplet, \
                            make_input_RSS, make_RSS, imsave, make_cut_irt

from metrics.new_1d_ssim import SSIM
from metrics.custom_losses import psnr, nmse

from data.data_transforms import root_sum_of_squares, kspace_to_nchw, apply_retro_mask, fake_input_gen, \
    ifft2, pad_FCF, complex_center_crop, center_crop, complex_height_crop, fake_input_gen_rss, nchw_to_kspace, fft2, \
    k_slice_to_chw, chw_to_k_slice, kspace_transform, normalize_im

from scipy.io import loadmat, savemat


class ModelTrainer:
    """
    Model Trainer for k-space learning or complex image learning
    with losses in complex image domains and real valued image domains.
    All learning occurs in k-space or complex image domains
    while all losses are obtained from either complex images or real-valued images.
    """

    def __init__(self, args, model, optimizer, train_loader, val_loader,
                 input_train_transform, input_val_transform, output_transform, losses, scheduler=None):

        multiprocessing.set_start_method(method='spawn')

        self.logger = get_logger(name=__name__, save_file=args.log_path / args.run_name)

        # Checking whether inputs are correct.
        assert isinstance(model, nn.Module), '`model` must be a Pytorch Module.'
        assert isinstance(optimizer, optim.Optimizer), '`optimizer` must be a Pytorch Optimizer.'
        assert isinstance(train_loader, DataLoader) and isinstance(val_loader, DataLoader), \
            '`train_loader` and `val_loader` must be Pytorch DataLoader objects.'

        assert callable(input_train_transform) and callable(input_val_transform), \
            'input_transforms must be callable functions.'
        # I think this would be best practice.
        assert isinstance(output_transform, nn.Module), '`output_transform` must be a Pytorch Module.'

        # 'losses' is expected to be a dictionary.
        losses = nn.ModuleDict(losses)

        # Display interval of 0 means no display of validation images on TensorBoard.
        if args.display_images <= 0:
            self.display_interval = 0
        else:
            self.display_interval = int(len(val_loader.dataset) // (args.display_images * args.batch_size))

        self.checkpointer = CheckpointManager(model, optimizer, mode='min',
                                              save_best_only=args.save_best_only, ckpt_dir=args.ckpt_path,
                                              max_to_keep=args.max_to_keep)

        # loading from checkpoint if specified.
        if vars(args).get('load_ckpt'):
            self.checkpointer.load(args.prev_model_ckpt_G, args.prev_model_ckpt_D, load_optimizer=False)

        self.name = args.name
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.input_train_transform = input_train_transform
        self.input_val_transform = input_val_transform
        self.output_transform = output_transform
        self.losses = losses
        self.scheduler = scheduler

        self.verbose = args.verbose
        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size
        self.use_slice_metrics = args.use_slice_metrics
        self.writer = SummaryWriter(str(args.log_path))
        self.ssim = SSIM(filter_size=7).to(device=args.device)

        self.use_residual = args.use_residual
        self.patch_size = args.patch_size

    def train_model(self):
        tic_tic = time()
        self.logger.info(self.name)
        self.logger.info('Beginning Training Loop.')
        for epoch in range(1, self.num_epochs + 1):  # 1 based indexing
            # Training
            tic = time()
            train_epoch_loss, train_epoch_metrics = self._train_epoch(epoch=epoch)
            toc = int(time() - tic)
            self._log_epoch_outputs(epoch, train_epoch_loss,
                                    train_epoch_metrics, elapsed_secs=toc, training=True, verbose=True)

            # Validation
            tic = time()
            val_epoch_loss, val_epoch_metrics = self._val_epoch(epoch=epoch)
            toc = int(time() - tic)
            self._log_epoch_outputs(epoch, val_epoch_loss, val_epoch_metrics,
                                        elapsed_secs=toc, training=False, verbose=True)

            self.checkpointer.save(metric=val_epoch_loss, verbose=True)

            if self.scheduler is not None:
                self.scheduler.step()

        # Finishing Training Loop
        self.writer.close()  # Flushes remaining data to TensorBoard.
        toc_toc = int(time() - tic_tic)
        self.logger.info(f'Finishing Training Loop. Total elapsed time: '
                         f'{toc_toc // 3600} hr {(toc_toc // 60) % 60} min {toc_toc % 60} sec.')

    def _train_epoch(self, epoch):
        self.model.train()
        torch.autograd.set_grad_enabled(True)

        epoch_loss = list()
        epoch_metrics = defaultdict(list)

        data_loader = enumerate(self.train_loader, start=1)
        if not self.verbose:  # tqdm has to be on the outermost iterator to function properly.
            data_loader = tqdm(data_loader, total=len(self.train_loader.dataset) / self.batch_size)

        # 'targets' is a dictionary containing k-space targets, cmg_targets, and img_targets.
        for step, data in data_loader:
            with torch.no_grad():  # Data pre-processing should be done without gradients.
                inputs, targets, input_fname, target_fname, extra_params = self.input_train_transform(*data)
            # For debugging purpose
            # tmp_input = inputs[0, :, 64, :, :]
            # tmp_target = targets[0, :, 64, :, :]
            # tmp_input = normalize_im(tmp_input.detach().cpu())
            # tmp_target = normalize_im(tmp_target.detach().cpu())
            # tmp_input = tmp_input.permute(1, 2, 0).numpy()
            # tmp_target = tmp_target.permute(1, 2, 0).numpy()
            # plt.imshow(tmp_input)
            # plt.savefig("input.png")
            # plt.imshow(tmp_target)
            # plt.savefig("label.png")
            # import ipdb;
            # ipdb.set_trace()


            recons, step_loss, step_metrics = self._train_step(inputs, targets, input_fname, target_fname, extra_params)
            epoch_loss.append(step_loss.detach())

            with torch.no_grad():  # Update epoch loss and metrics
                if self.use_slice_metrics:
                    slice_metrics = self._get_slice_metrics(recons, targets, self.batch_size)
                    step_metrics.update(slice_metrics)
                [epoch_metrics[key].append(value.detach()) for key, value in step_metrics.items()]

                if self.verbose:
                    self._log_step_outputs(epoch, step, step_loss, step_metrics, training=True)

                x_irt_grid = make_cut_irt(inputs, recons, targets, dim=-3)
                y_irt_grid = make_cut_irt(inputs, recons, targets, dim=-2)
                z_irt_grid = make_cut_irt(inputs, recons, targets, dim=-1)

                self.writer.add_image(f'Train_irt_x/{step}', x_irt_grid, epoch, dataformats='HWC')
                self.writer.add_image(f'Train_irt_y/{step}', y_irt_grid, epoch, dataformats='HWC')
                self.writer.add_image(f'Train_irt_z/{step}', z_irt_grid, epoch, dataformats='HWC')

        # Converted to scalar and dict with scalar forms.
        return self._get_epoch_outputs(epoch, epoch_loss, epoch_metrics, training=True)

    def _train_step(self, inputs, targets, input_fname, target_fname, extra_params):
        self.optimizer.zero_grad()

        outputs = self.model(inputs) ## 60 channel output
        # Residual learning scheme
        if self.use_residual:
            outputs = outputs + inputs
        recons = self.output_transform(outputs, extra_params)

        mse_loss = self.losses['mse_loss'](recons, targets)
        step_metrics = {'mse_loss': mse_loss}

        step_loss = mse_loss
        step_loss.backward()

        self.optimizer.step()

        return recons, step_loss, step_metrics

    def _val_epoch(self, epoch):
        self.model.eval()
        torch.autograd.set_grad_enabled(False)

        epoch_loss = list()
        epoch_metrics = defaultdict(list)

        data_loader = enumerate(self.val_loader, start=1)
        if not self.verbose:
            data_loader = tqdm(data_loader, total=len(self.val_loader.dataset)/self.batch_size)

        # 'targets' is a dictionary containing k-space targets, cmg_targets, and img_targets.
        for step, data in data_loader:
            inputs, targets, input_fname, target_fname, extra_params = self.input_val_transform(*data)
            recons, step_loss, step_metrics = self._val_step(inputs, targets, input_fname, target_fname, extra_params)
            epoch_loss.append(step_loss.detach())

            if self.use_slice_metrics:
                slice_metrics = self._get_slice_metrics(recons, targets, self.batch_size)
                step_metrics.update(slice_metrics)

            [epoch_metrics[key].append(value.detach()) for key, value in step_metrics.items()]

            if self.verbose:
                self._log_step_outputs(epoch, step, step_loss, step_metrics, training=False)

            # Save images to TensorBoard.
            # Condition ensures that self.display_interval != 0 and that the step is right for display.
            x_irt_grid = make_cut_irt(inputs, recons, targets, dim=-3)
            y_irt_grid = make_cut_irt(inputs, recons, targets, dim=-2)
            z_irt_grid = make_cut_irt(inputs, recons, targets, dim=-1)

            self.writer.add_image(f'Val_irt_x/{step}', x_irt_grid, epoch, dataformats='HWC')
            self.writer.add_image(f'Val_irt_y/{step}', y_irt_grid, epoch, dataformats='HWC')
            self.writer.add_image(f'Val_irt_z/{step}', z_irt_grid, epoch, dataformats='HWC')

        epoch_loss, epoch_metrics = self._get_epoch_outputs(epoch, epoch_loss, epoch_metrics, training=False)
        return epoch_loss, epoch_metrics

    def _val_step(self, inputs, targets, input_fname, target_fname, extra_params):
        outputs = self.model(inputs)
        if self.use_residual:
            outputs += inputs
        recons = self.output_transform(outputs, extra_params)

        mse_loss = self.losses['mse_loss'](recons, targets)

        step_metrics = {'mse_loss': mse_loss}
        step_loss = mse_loss

        return recons, step_loss, step_metrics

    def _get_slice_metrics(self, recons, targets, batch_size):
        img_recons = recons.squeeze().detach()  # Just in case.
        img_targets = targets.squeeze().detach()

        if batch_size != 1:
            slice_ssim = 0
            slice_psnr = 0
            slice_nmse = 0
            for i in range(batch_size):
                max_range = img_targets.max() - img_targets.min()
                slice_ssim += self.ssim(img_recons, img_targets)
                slice_psnr += psnr(img_recons, img_targets, data_range=max_range)
                slice_nmse += nmse(img_recons, img_targets)
            slice_ssim /= batch_size
            slice_psnr /= batch_size
            slice_nmse /= batch_size
        else:  # When single batch is implemented

            max_range = img_targets.max() - img_targets.min()
            slice_ssim = self.ssim(img_recons, img_targets)
            slice_psnr = psnr(img_recons, img_targets, data_range=max_range)
            slice_nmse = nmse(img_recons, img_targets)

        slice_metrics = {
            'slice/ssim': slice_ssim,
            'slice/nmse': slice_nmse,
            'slice/psnr': slice_psnr
        }

        return slice_metrics

    def _get_epoch_outputs(self, epoch, epoch_loss, epoch_metrics, training=True):
        mode = 'Training' if training else 'Validation'
        num_slices = len(self.train_loader.dataset) if training else len(self.val_loader.dataset)

        # Checking for nan values.
        epoch_loss = torch.stack(epoch_loss)
        is_finite = torch.isfinite(epoch_loss)
        num_nans = (is_finite.size(0) - is_finite.sum()).item()

        if num_nans > 0:
            self.logger.warning(f'Epoch {epoch} {mode}: {num_nans} NaN values present in {num_slices} slices')
            epoch_loss = torch.mean(epoch_loss[is_finite]).item()
        else:
            epoch_loss = torch.mean(epoch_loss).item()

        for key, value in epoch_metrics.items():
            epoch_metric = torch.stack(value)
            is_finite = torch.isfinite(epoch_metric)
            num_nans = (is_finite.size(0) - is_finite.sum()).item()

            if num_nans > 0:
                self.logger.warning(f'Epoch {epoch} {mode} {key}: {num_nans} NaN values present in {num_slices} slices')
                epoch_metrics[key] = torch.mean(epoch_metric[is_finite]).item()
            else:
                epoch_metrics[key] = torch.mean(epoch_metric).item()

        return epoch_loss, epoch_metrics

    def _log_step_outputs(self, epoch, step, step_loss, step_metrics, training=True):
        mode = 'Training' if training else 'Validation'
        self.logger.info(f'Epoch {epoch:03d} Step {step:03d} {mode} loss: {step_loss.item():.4e}')
        for key, value in step_metrics.items():
            self.logger.info(f'Epoch {epoch:03d} Step {step:03d}: {mode} {key}: {value.item():.4e}')

    def _log_epoch_outputs(self, epoch, epoch_loss, epoch_metrics,
                           elapsed_secs, training=True, verbose=True):
        mode = 'Training' if training else 'Validation'
        self.logger.info(f'Epoch {epoch:03d} {mode}. _loss: {epoch_loss:.4e},'
                         f'Time: {elapsed_secs // 60} min {elapsed_secs % 60} sec')
        self.writer.add_scalar(f'{mode}_epoch_loss', scalar_value=epoch_loss, global_step=epoch)

        if verbose:
            for key, value in epoch_metrics.items():
                self.logger.info(f'Epoch {epoch:03d} {mode}. {key}: {value:.4e}')
                self.writer.add_scalar(f'{mode}_epoch_{key}', scalar_value=value, global_step=epoch)
