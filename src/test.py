""" StarGAN v2 """
import os
import warnings
warnings.filterwarnings("ignore")
import math
import argparse
import numpy as np
from imageio import imsave
from PIL import Image
import shutil

import torch
from torch.backends import cudnn
from torchvision import transforms as T
import torch.nn.functional as F

from core.solver import Solver
from core.checkpoint import CheckpointIO
import core.utils as utils

from utilgan import img_list, dir_list, basename
try: # progress bar for notebooks 
    get_ipython().__class__.__name__
    from progress_bar import ProgressIPy as ProgressBar
except: # normal console
    from progress_bar import ProgressBar

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--source', default=None, help='Source image or directory with images')
parser.add_argument('--out_dir', default='_out', help='Directory to save generated images and videos')
parser.add_argument('--refs', default=None, help='String of class indices, separated by "-", a la 0-0-10-20-..')
parser.add_argument('--model', default='models', help='Saved network checkpoint')
parser.add_argument('--lowmem', action='store_true', help="Aggressive memory cleanup for Generator")
parser.add_argument('--parallel', action='store_true', help="Parallel processing?")
parser.add_argument('--gpu', type=int, default=0, help="Which GPU to use")
parser.add_argument('--seed', type=int, default=None)
args = parser.parse_args()

@torch.no_grad()
def main():
    cudnn.benchmark = True
    device = torch.device('cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu')
    args.mode = 'process'
    os.makedirs(args.out_dir, exist_ok=True)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    # dirty hacks to obtain settings from saved model
    test_net = torch.load(args.model)
    test_gen = test_net['generator']
    test_mod = test_net['mapping_network']
    last_decode = [key for key in test_gen.keys() if 'decode' in key][-1]
    args.img_size = 2 ** (int(last_decode.split('.')[1]) + 3)
    last_unshared_layer = [key for key in test_mod.keys() if 'unshared' in key][-1]
    args.num_domains = int(last_unshared_layer.split('.')[1]) + 1
    mod_shapes = [test_mod[key].shape for key in test_mod.keys()]
    args.latent_dim = mod_shapes[0][-1]  # mapping input
    args.style_dim  = mod_shapes[-1][-1] # mapping output
    del test_net, test_gen, test_mod
    
    args.model_dir = os.path.relpath(os.path.abspath(os.path.dirname(args.model)))
    args.model = basename(args.model)

    solver = Solver(args, device)
    nets_ema = solver.nets_ema
    ckpt = CheckpointIO(args.model_dir, args.model, None, **nets_ema)
    ckpt.load()

    transform_src = T.Compose([T.ToTensor(), T.Normalize([.5,.5,.5],[.5,.5,.5]),])
    transform_ref = T.Compose([T.ToTensor(), T.Normalize([.5,.5,.5],[.5,.5,.5]),T.Resize([args.img_size, args.img_size]),])
    
    # make styles from refs/classes
    if args.refs is not None and os.path.exists(args.refs): # file(s) as ref(s)
        refs = img_list(args.refs) if os.path.isdir(args.refs) else [args.refs]
        print(' loading %d refs' % len(refs))
        cls_names = [basename(fn) for fn in refs]
        cls_refs = [int(fn[0]) for fn in cls_names] # first char in ref name must be class num!
        cls_refs = torch.tensor(cls_refs).to(device) # [n]
        img_refs = [Image.open(fname).convert('RGB') for fname in refs]
        img_refs = torch.cat([transform_ref(img).unsqueeze(0).to(device) for img in img_refs]) # [n,c,h,w]
        styles = nets_ema.style_encoder(img_refs, cls_refs).unsqueeze(1) # [n,1,64]
    else:
        if args.refs is not None: # string with class numbers
            cls_refs = [int(cls) for cls in args.refs.split('-')]
        else: # random
            cls_refs = list(range(args.num_domains))
        cls_refs = torch.tensor(cls_refs).to(device) # [n]
        cls_names = [int(c) for c in cls_refs]
        z_latents = torch.randn(len(cls_names), args.latent_dim).to(device) # [n,16]
        styles = nets_ema.mapping_network(z_latents, cls_refs).unsqueeze(1) # [n,1,64]
    stylecount = styles.shape[0]

    def proc_img(img, style):
        img_out = nets_ema.generator(img, style).cpu()
        return utils.post_np(img_out)[0]

    # process all source images
    srcs = img_list(args.source) if os.path.isdir(args.source) else [args.source]
    pbar = ProgressBar(len(srcs) * stylecount)
    for i, img_path in enumerate(srcs):
        img_src = Image.open(img_path).convert('RGB')
        img_src = transform_src(img_src).to(device).unsqueeze(0)
        file_out = os.path.join(args.out_dir, '%s-%s' % (basename(img_path), args.model))

        for i in range(stylecount):
            img_out = proc_img(img_src, styles[i])
            imsave(file_out + '-%s.jpg' % str(cls_names[i]), img_out)
            pbar.upd()


if __name__ == '__main__':
    main()
