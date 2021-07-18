""" StarGAN v2  Copyright (c) 2020-present NAVER Corp.
http://creativecommons.org/licenses/by-nc/4.0
"""
import os
import warnings
warnings.filterwarnings("ignore")
import argparse
from munch import Munch
from PIL import Image

import torch
from torch.backends import cudnn
from torchvision import transforms

from core.data_loader import get_train_loader
from core.solver import Solver

from utilgan import save_cfg, basename

parser = argparse.ArgumentParser()
# main args
parser.add_argument('--data_dir', default='data', help='Directory containing training images')
parser.add_argument('--model_dir', default='train', help='Directory for saving network checkpoints')
parser.add_argument('-r', '--resume', type=int, default=0, help='K/iterations to resume training/testing (in thousands)')
parser.add_argument('--lowmem', action='store_true', help="Aggressive memory cleanup for Generator")
parser.add_argument('--parallel', action='store_true', help="Parallel processing?")
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers used in DataLoader')
parser.add_argument('--gpu', type=int, default=0, help="Which GPU to use")
# model args
parser.add_argument('--img_size', type=int, default=256, help='Image resolution')
parser.add_argument('--latent_dim', type=int, default=16, help='Latent vector dimension')
parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension of mapping network')
parser.add_argument('--style_dim', type=int, default=64, help='Style code dimension')
# weights for objective functions
parser.add_argument('--lambda_ds', type=float, default=2, help='Weight for diversity sensitive loss')
parser.add_argument('--lambda_cyc', type=float, default=1, help='Weight for cyclic consistency loss')
parser.add_argument('--lambda_sty', type=float, default=1, help='Weight for style reconstruction loss')
parser.add_argument('--lambda_reg', type=float, default=1, help='Weight for R1 regularization')
# training args
parser.add_argument('--batch', type=int, default=3, help='Batch size for training') # 8
parser.add_argument('--val_batch', type=int, default=4, help='Batch size for validation') # 32
parser.add_argument('--total_iters', type=int, default=100000, help='Number of total iterations')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for D, E and G')
parser.add_argument('--f_lr', type=float, default=1e-6, help='Learning rate for F')
parser.add_argument('--beta1', type=float, default=0.0, help='Decay rate for 1st moment of Adam')
parser.add_argument('--beta2', type=float, default=0.99, help='Decay rate for 2nd moment of Adam')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
parser.add_argument('--sample_per_domain', type=int, default=1, help='Number of generated images per domain during sampling') # 10
parser.add_argument('--seed', type=int, default=777, help='Seed for random number generator')
# step size
parser.add_argument('--log_every', type=int, default=10)
parser.add_argument('--sample_every', type=int, default=1000) # 5000
parser.add_argument('--save_every', type=int, default=5000) # 10000
args = parser.parse_args()

args.mode = 'train'
args.num_workers = min(args.num_workers, args.batch)
args.ds_iter = args.total_iters * 2 # Number of iterations to optimize diversity sensitive loss

def subdirs(dname):
    return sorted([d for d in os.listdir(dname) if os.path.isdir(os.path.join(dname, d))])

def main():

    device = torch.device('cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu')
    print(device, 'batch', args.batch)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)

    assert len(os.listdir(args.data_dir)) > 0, "No data at %s" % args.data_dir
    domains = subdirs(args.data_dir)
    args.num_domains = len(domains)
    args.domains = [basename(dom) for dom in domains]
    save_cfg(args, args.model_dir)
        
    solver = Solver(args, device)

    args.sample_dir = os.path.join(args.model_dir, 'test')
    loaders = Munch(src=get_train_loader(root=args.data_dir, which='src', img_size=args.img_size, batch_size=args.batch, num_workers=args.num_workers),
                    ref=get_train_loader(root=args.data_dir, which='ref', img_size=args.img_size, batch_size=args.batch, num_workers=args.num_workers),
                    val=get_train_loader(root=args.data_dir, which='val', img_size=args.img_size, batch_size=args.val_batch, num_workers=args.num_workers))
    solver.train(loaders)


if __name__ == '__main__':
    main()
