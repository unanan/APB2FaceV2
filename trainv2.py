import os
import time
import argparse
from trainerv2.trainerv2 import TrainerV2
from util.logger import get_logger, show_opt

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'y', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'n', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_options():
    parser = argparse.ArgumentParser(description="APB2FaceV2 Training V2")

    # logging
    parser.add_argument('--checkpoint', type=str,       default='./checkpoints',    help='experimental results are saved here')
    parser.add_argument('--name',       type=str,       default='AnnVI',            help='experimental name')

    # data
    parser.add_argument('--data',       type=str,       default='AnnVI',            help='AnnVI | VoxCeleb2')
    parser.add_argument('--data_root',  type=str,       default="/usr/stable/apb/raw/AnnVI/feature")
    parser.add_argument('--img_size',   type=int,       default=256,                help='image size')
    parser.add_argument('--mode',       type=str,       default='train', choices=TrainerV2.MODES, help='train | test')
    # model
    parser.add_argument('--trainer',    type=str,       default='l2face',           help='chooses which trainer to use.')
    parser.add_argument('--init_type',  type=str,       default='normal',           help='normal | xavier | kaiming | orthogonal')
    parser.add_argument('--gan_mode',   type=str,       default='lsgan',            help='vanilla | lsgan | wgangp')
    # trainbeta1
    parser.add_argument('--gpus',       type=str,       default='0',                help='e.g. [-1:CPU | 0:gpu0 | 0,1:gpu 0 and 1]')
    parser.add_argument('--optim',      type=str,       default='Adam',             help='Adam | SGD | RMSProp')
    parser.add_argument('--worker_number', type=int,    default=4,                  help='worker number for loading data')
    parser.add_argument('--batch_size', type=int,       default=16,                 help='input batch size')
    parser.add_argument('--show_iters', type=int,       default=200,                help='iteration')

    parser.add_argument('--batch_id',   type=int,       default=4,                  help='id number')
    parser.add_argument('--batch_img',  type=int,       default=3,                  help='image number for one id')
    # test
    parser.add_argument('--idt_test',   type=str,       default='random',           help='which identity to test')
    parser.add_argument('--own_audio',  type=str2bool,  default=True,               help='whether use onw audio')
    parser.add_argument('--non_ref',    type=str2bool,  default=False,
                        help='whether use black image as the reference image')

    parser.add_argument('--lr',         type=float,     default=0.0002,             help='initial learning rate for adam')
    parser.add_argument('--lr_policy',  type=str,       default='linear',           help='linear | step | plateau | cosine')
    parser.add_argument('--lr_decay_iters', type=int,   default=50,
                        help='multiply by a gamma every lr_decay_iters iterations')
    parser.add_argument('--niter',      type=int,       default=30,                 help='# of iter at starting learning rate')
    parser.add_argument('--niter_decay', type=int,      default=40,                 help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('--beta1',      type=float,     default=0.5,                help='momentum term of adam')
    parser.add_argument('--momentum',   type=float,     default=0.9,                help='momentum term of SGD')

    parser.add_argument('--record_every', type=int,     default=5,                  help='every # epochs to record the checkpoint once')
    parser.add_argument('--resume', '-r', type=bool,    default=False,              help='resume')
    parser.add_argument('--resume_name', type=str,      default='',                 help='resume name')
    parser.add_argument('--resume_epoch', type=int,     default=None,               help='resume epoch')

    opt = parser.parse_args()
    # modify parser

    opt.gpus = [int(dev) for dev in opt.gpus.split(',')]
    os.makedirs(opt.checkpoint, exist_ok=True)

    # init checkpoint
    if opt.resume:
        opt.name_dir = opt.resume_name
        opt.logdir = '{}/{}'.format(opt.checkpoint, opt.name_dir)
    else:
        opt.name_dir = '{}-{}'.format(opt.name, time.strftime("%Y%m%d-%H%M%S"))
        opt.logdir = '{}/{}'.format(opt.checkpoint, opt.name_dir)
        os.makedirs(opt.logdir, exist_ok=True)
    return opt


if __name__ == '__main__':
    opt = get_options()
    logger = get_logger(opt)
    show_opt(opt, logger)


    trainer = TrainerV2(opt, logger)
    trainer.train()
