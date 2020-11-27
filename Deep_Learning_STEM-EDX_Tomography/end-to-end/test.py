import torch
from torch import nn, optim
from pathlib import Path

from utils.run_utils import initialize, save_dict_as_json, get_logger, create_arg_parser
from utils.train_utils import create_inference_data_loaders

from data.input_transforms import PreProcessInfer_avg
from data.output_transforms import OutputScalePatchTransform_avg

from pytorch3dunet.unet3d.model import UNet3D
from eval_scripts.Inference_3D_avg import Infer


def train_img(args):

    # Creating checkpoint and logging directories, as well as the run name.
    ckpt_path = Path(args.ckpt_root)
    ckpt_path.mkdir(exist_ok=True)

    ckpt_path = ckpt_path
    ckpt_path.mkdir(exist_ok=True)

    run_number, run_name = initialize(ckpt_path)

    ckpt_path = ckpt_path / run_name
    ckpt_path.mkdir(exist_ok=True)

    log_path = Path(args.log_root)
    log_path.mkdir(exist_ok=True)

    log_path = log_path
    log_path.mkdir(exist_ok=True)

    log_path = log_path / run_name
    log_path.mkdir(exist_ok=True)

    logger = get_logger(name=__name__, save_file=log_path / run_name)

    # Assignment inside running code appears to work.
    if (args.gpu is not None) and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        logger.info(f'Using GPU {args.gpu} for {run_name}')
    else:
        device = torch.device('cpu')
        logger.info(f'Using CPU for {run_name}')

    # Saving peripheral variables and objects in args to reduce clutter and make the structure flexible.
    args.run_number = run_number
    args.run_name = run_name
    args.ckpt_path = ckpt_path
    args.log_path = log_path
    args.device = device

    save_dict_as_json(vars(args), log_dir=log_path, save_name=run_name)

    # UNET architecture requires that all inputs be dividable by some power of 2.
    divisor = 2 ** args.num_pool_layers

    eval_input_transform = PreProcessInfer_avg(args.device, use_seed=False, divisor=divisor, patch_size=args.patch_size,
                                               stride=args.stride)

    # DataLoaders
    eval_loader = create_inference_data_loaders(args)

    eval_output_transform = OutputScalePatchTransform_avg(patch_size=args.patch_size, stride=args.stride)

    data_chans = 3
    out_chans = 3

    model = UNet3D(data_chans, out_chans, final_sigmoid=False, f_maps=args.chans, num_levels=args.num_pool_layers).to(
        device)

    if args.load_ckpt:
        model_ckpt = args.prev_model_ckpt
        save_dict = torch.load(model_ckpt)
        model.load_state_dict(save_dict['model_state_dict'])
        print('Loaded model checkpoint')

    trainer = Infer(args, model, eval_loader, eval_input_transform, eval_output_transform)

    trainer.inference_patchvol(args)


if __name__ == '__main__':

    settings = dict(
        # Variables that almost never change.
        name='test',
        data_root='./datasets',
        log_root='./logs',
        ckpt_root='./checkpoints',
        batch_size=1,
        chans=32,
        num_pool_layers=3,
        save_best_only=True,

        # Variables that occasionally change.
        num_workers=0,
        gpu=0,  # Set to None for CPU mode.
        use_residual=True,
        # For avg, patch size and stride should be given as list of tuples
        patch_size=[
            (256, 256, 32),
            (256, 32, 256),
            (32, 256, 256)],
        stride=[
            (0, 0, 16),
            (0, 16, 0),
            (16, 0, 0)],
        iter=3,

        # Prev model ckpt
        load_ckpt=True,
        prev_model_ckpt='./baseline_residual_13view/ckpt_150.tar',

        # Variables that change frequently.
        verbose=False,
        use_slice_metrics=True,  # Using slice metrics causes a 30% increase in training time.

        # Evaluation,
        save_fdir='./results',
    )
    options = create_arg_parser(**settings).parse_args()
    train_img(options)