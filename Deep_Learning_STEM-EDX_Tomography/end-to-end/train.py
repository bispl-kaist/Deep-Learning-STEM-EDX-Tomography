import torch
from torch import nn, optim
from pathlib import Path

from utils.run_utils import initialize, save_dict_as_json, get_logger, create_arg_parser
from utils.train_utils import create_data_loaders

from data.input_transforms import Prefetch2Device, PreProcessScale_random
from data.output_transforms import SingleOutputTransform

from pytorch3dunet.unet3d.model import UNet3D
from train.model_trainers.trainer_3D import ModelTrainer


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

    # Input transforms. These are on a per-slice basis.
    # UNET architecture requires that all inputs be dividable by some power of 2.
    divisor = 2 ** args.num_pool_layers

    data_prefetch = Prefetch2Device(device)

    input_train_transform = PreProcessScale_random(args.device, use_seed=False, divisor=divisor, patch_size=args.patch_size)
    input_val_transform = PreProcessScale_random(args.device, use_seed=False, divisor=divisor, patch_size=args.patch_size)

    # DataLoaders
    train_loader, val_loader = create_data_loaders(args, transform=data_prefetch)

    losses = dict(
        mse_loss=nn.MSELoss(reduction='mean'),
    )

    output_transform = SingleOutputTransform()

    data_chans = 3
    out_chans = 3

    model = UNet3D(data_chans, out_chans, final_sigmoid=False, f_maps=args.chans, num_levels=args.num_pool_layers).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_red_epoch, gamma=args.lr_red_rate)

    trainer = ModelTrainer(args, model, optimizer, train_loader, val_loader, input_train_transform, input_val_transform,
                           output_transform, losses, scheduler)
    trainer.train_model()

if __name__ == '__main__':
    settings = dict(
        name='baseline_residual_128',
        data_root='/media/harry/mri1/backup/SAIT_hj/datasets/SRTOMO_13view_postSRCNN',
        log_root='./logs',
        ckpt_root='./checkpoints',
        batch_size=1,
        chans=32,
        num_pool_layers=3,
        save_best_only=False,

        # Variables that occasionally change.
        display_images=1,  # Maximum number of images to save.
        num_workers=3,
        init_lr=1e-4,
        gpu=0,  # Set to None for CPU mode.
        max_to_keep=10,
        use_residual=True,

        start_slice=1,
        start_val_slice=1,
        patch_size=[(128, 128, 128),
                    (256, 256, 32),
                    (256, 32, 256),
                    (32, 256, 256)],

        # Prev model ckpt
        load_ckpt=False,
        prev_model_ckpt_G=None,
        prev_model_ckpt_D=None,

        # Variables that change frequently.
        num_epochs=150,
        verbose=False,
        use_slice_metrics=True,  # Using slice metrics causes a 30% increase in training time.
        lr_red_epoch=100,
        lr_red_rate=0.1,

        # Evaluation
        eval_fdir='./test_axial/val_input/',
    )
    options = create_arg_parser(**settings).parse_args()
    train_img(options)