import torch
import yaml
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
import warnings
warnings.filterwarnings('ignore')

from model import MInterface
from data import DInterface
from utils import load_model_path_by_args


def load_callbacks(args):
    callbacks = []

    # model auto-save
    callbacks.append(plc.ModelCheckpoint(
        monitor='loss',
        dirpath=args.checkpoint_root,
        filename='best-stage=%01d-cycle=%02d-{epoch:03d}-{loss:.2f}' %(args.training_stage, args.cycle),
        save_top_k=1,
        mode='min',
        save_last=True
    ))

    if args.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(
            logging_interval='epoch'))
    return callbacks


def first_stage_training(args):
    pl.seed_everything(args.seed)
    data_module = DInterface(**vars(args))
    model = MInterface(**vars(args))

    args.callbacks = load_callbacks(args)
    trainer = Trainer.from_argparse_args(args, accelerator='gpu', devices=[0,1])

    trainer.fit(model, data_module)

    return args.callbacks[0].best_model_path

def second_stage_training(args):
    pl.seed_everything(args.seed)
    data_module = DInterface(**vars(args))
    model = MInterface(**vars(args))

    args.callbacks = load_callbacks(args)
    trainer = Trainer.from_argparse_args(args, accelerator='gpu', devices=[0,1])

    trainer.fit(model, data_module)

    return args.callbacks[0].best_model_path

def test(args):
    pl.seed_everything(args.seed)
    load_path = load_model_path_by_args(args)
    data_module = DInterface(**vars(args))
    
    model = MInterface(**vars(args))

    if load_path is not None:
        args.ckpt_path = load_path
        model = model.load_from_checkpoint(args.ckpt_path, **vars(args))
        print("Load model from ", args.ckpt_path)

    trainer = Trainer.from_argparse_args(args, accelerator='gpu', devices=[0])

    trainer.test(model, data_module)
    
if __name__ == '__main__':
    parser = ArgumentParser()
    
    parser.add_argument('--yaml_path', default=None, help="load params from yaml file")

    # Basic Training Control
    parser.add_argument('--batch_size', default=4, type=int, help="batch size")
    parser.add_argument('--num_workers', default=4, type=int, help="number of workers")
    parser.add_argument('--seed', default=1226, type=int, help="random seed")
    parser.add_argument('--lr', default=1e-3, type=float, help="initial learning rate")

    # LR Scheduler
    parser.add_argument('--lr_scheduler', default='cosine', type=str, help="learning rate scheduler")
    parser.add_argument('--lr_decay_steps', default=200, type=int, help="learning rate decay steps")
    parser.add_argument('--lr_decay_rate', default=0.5, type=float, help="learning rate decay rate")
    parser.add_argument('--lr_decay_min_lr', default=1e-5, type=float, help="minimal learning rate")
    parser.add_argument('--warmup_epoch', default=10, type=int, help="number of warmup epoch")

    # Dataset Info
    parser.add_argument('--dataset', default='deblur_data', type=str, help="dataset name")
    parser.add_argument('--root_path', default='datasets/MS-RBD', type=str, help="dataset path")
    parser.add_argument('--num_bins', default=16, type=int, help="number of event bins")
    parser.add_argument('--roi_size', default=(64,64), type=tuple, help="roi size for training")
    parser.add_argument('--scale_factor', default=4, type=int, help="scale ratio of images over events")
    parser.add_argument('--train', action='store_true', help="training mode or test mode")

    # Training Info
    parser.add_argument('--checkpoint_root', default='checkpoint/MS-RBD', type=str, help="path to save checkpoints")
    parser.add_argument('--loss', default='l1', type=str, help="loss function")
    parser.add_argument('--weight_decay', default=0, type=float, help="weight decay")
    parser.add_argument('--first_stage_epoch', default=210, type=int, help="epoch for first-stage training")
    parser.add_argument('--second_stage_cycle', default=15, type=int, help="cycle for second-stage training")
    parser.add_argument('--second_stage_epoch', default=30, type=int, help="epoch for second-stage training")
    parser.add_argument('--loss_weight', default=[50, 1, 50, 50], type=list, help="balancing weights for [loss_BC, loss_SC, loss_TG, loss_SG]") 

    # Test Info
    parser.add_argument('--save_path', default='results/MS-RBD', type=str, help="path to save results")
    parser.add_argument('--load_dir', default='checkpoint/MS-RBD', type=str, help="path to load checkpoints")
    parser.add_argument('--load_stage', default=2, type=int, help="choose model of which stage to load")
    parser.add_argument('--load_cycle', default=15, type=int, help="choose model of which cycle to load")
    parser.add_argument('--has_gt', action='store_true', help="set ture to compute metrics (PSNR and SSIM)")
    parser.add_argument('--predict_ts', default=0.5, type=float, help="predict the image at the normalized target ts in [0,1]")

    # Model Hyperparameters
    parser.add_argument('--model_name', default='san', type=str, help="network name")
    parser.add_argument('--im_in_channel', default=3, type=int, help="image input channel")
    parser.add_argument('--out_channel', default=3, type=int, help="output channel")
    parser.add_argument('--ev_in_channel', default=96, type=int, help="event input channel")
    parser.add_argument('--initializor', default='kaiming', type=str, help="model parameter initializor")

    # Add pytorch lightning's args to parser as a group.
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # Load parameters from YAML file
    if args.yaml_path:
        yaml_file = args.yaml_path
        with open(yaml_file, 'r') as file:
            yaml_data = yaml.safe_load(file)
        for key, value in yaml_data.items():
            setattr(args, key, value)

    if args.train == True: 
        # for training
        torch.multiprocessing.set_sharing_strategy('file_system') 
        args.default_root_dir = args.checkpoint_root

        # first stage training
        args.training_stage = 1
        args.cycle = 0
        args.max_epochs = args.first_stage_epoch 
        best_model_path = first_stage_training(args)

        # second stage training
        for cycle in range(1, args.second_stage_cycle+1):

            print("Begin second stage training, cycle ", cycle, "==================")
            args.training_stage = 2
            args.cycle = cycle
            args.warmup_epoch = 0
            args.lr = 5e-4
            args.max_epochs = args.second_stage_epoch
            args.lr_decay_steps = args.second_stage_epoch
            args.ckpt_path = best_model_path
            print("Load model from ", args.ckpt_path)

            best_model_path = second_stage_training(args)
    
    else:
        ##for test
        test(args)


