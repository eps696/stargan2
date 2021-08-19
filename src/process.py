""" StarGAN v2 """
import os
import os.path as osp
# import warnings
# warnings.filterwarnings("ignore")
import math
import argparse
import random
import numpy as np
from imageio import imsave
from PIL import Image
import cv2

import torch
from torch.backends import cudnn
from torchvision import transforms
import torch.nn.functional as F
from torchvision import transforms as T

from core.solver import Solver
from core.checkpoint import CheckpointIO
import core.utils as utils

from utilgan import latent_anima, img_list, basename
try: # progress bar for notebooks 
    get_ipython().__class__.__name__
    from progress_bar import ProgressIPy as ProgressBar
except: # normal console
    from progress_bar import ProgressBar

parser = argparse.ArgumentParser()
# main
parser.add_argument('-i',  '--source',  default=None, help='Directory or file containing source images')
parser.add_argument('-o',  '--out_dir', default='_out', help='Directory to save generated images and videos')
parser.add_argument(       '--refs',    default=None, help='String of class indices, separated by "-", a la 0-0-10-20-..')
parser.add_argument(       '--model',   default='models', help='Saved network checkpoint')
# process 
parser.add_argument(       '--frames',  type=int, default=None, help="Total number of frames to render")
parser.add_argument(       '--fstep',   type=int, default=None, help="Number of frames for smooth interpolation")
parser.add_argument(       '--rand',    action='store_true', help="Randomize refs order (otherwise original)")
# misc
parser.add_argument(       '--lowmem',  action='store_true', help="Aggressive memory cleanup for Generator")
parser.add_argument(       '--parallel', action='store_true', help="Parallel processing?")
parser.add_argument(       '--gpu',     default=0, type=int, help="Which GPU to use")
parser.add_argument(       '--seed',    default=None, type=int)
a = parser.parse_args()

def checkout(img, i):
    filename = osp.join(a.out_dir, '%06d.jpg' % i)
    # imsave(filename, img, quality=95)
    cv2.imwrite(filename, img[:,:,::-1]) # faster
    
@torch.no_grad()
def main():
    cudnn.benchmark = False
    cudnn.deterministic = True
    device = torch.device('cuda:%d' % a.gpu if torch.cuda.is_available() else 'cpu')
    a.mode = 'process'
    a.out_dir = os.path.join(a.out_dir, basename(a.source))
    os.makedirs(a.out_dir, exist_ok=True)
    if a.seed is not None:
        torch.manual_seed(a.seed)
        np.random.seed(a.seed)

    # dirty hacks to obtain settings from saved model
    test_net = torch.load(a.model)
    test_gen = test_net['generator']
    test_map = test_net['mapping_network']
    last_decode = [key for key in test_gen.keys() if 'decode' in key][-1]
    a.img_size = 2 ** (int(last_decode.split('.')[1]) + 3)
    last_unshared_layer = [key for key in test_map.keys() if 'unshared' in key][-1]
    a.num_domains = int(last_unshared_layer.split('.')[1]) + 1
    mod_shapes = [test_map[key].shape for key in test_map.keys()]
    a.latent_dim = mod_shapes[0][-1]  # mapping input
    a.style_dim  = mod_shapes[-1][-1] # mapping output
    del test_net, test_gen, test_map
    
    a.model_dir = osp.relpath(osp.abspath(osp.dirname(a.model)))
    a.model = basename(a.model)

    # make network
    solver = Solver(a, device)
    # print(next(solver.parameters()).device) # double check if cuda
    nets_ema = solver.nets_ema
    ckpt = CheckpointIO(a.model_dir, a.model, None, **nets_ema)
    ckpt.load()
    transform_norm = T.Normalize([.5,.5,.5],[.5,.5,.5])
    transform_src = T.Compose([T.ToTensor(), T.Normalize([.5,.5,.5],[.5,.5,.5]),])
    transform_ref = T.Compose([T.ToTensor(), T.Normalize([.5,.5,.5],[.5,.5,.5]),T.Resize([a.img_size, a.img_size]),])

# # # load source[s]

    out_size = None
    srcs = img_list(a.source) if osp.isdir(a.source) else [a.source]
    src_count = len(srcs)
    if src_count == 1: # source = 1 pic
        img_src = Image.open(srcs[0]).convert('RGB')
        if out_size is None: out_size = img_src.size[::-1]
        img_src = transform_src(img_src).to(device).unsqueeze(0)
        if list(img_src.shape[-2:]) != out_size:
            img_src = F.interpolate(img_src, size=out_size, mode='bicubic', align_corners=True)
        framecount = a.frames
    else: # source = sequence
        framecount = src_count
    assert framecount is not None and framecount != 0, ' !! Undefined source framecount !! '

# # # load refs, make styles
    
    def gen_styles(refs):
        refs = torch.tensor(refs).to(device) # [b]
        keycount = refs.shape[0]
        z_latents = torch.randn(keycount, a.latent_dim).to(device) # [n,16]
        styles = nets_ema.mapping_network(z_latents, refs).cpu().numpy() # [n,64]
        for i in range(keycount):
            ref = str(refs[i].item())
        return styles # [n,64]

    # load refs and classes from files
    assert a.refs is not None, " Set domains to process by --refs argument"
    if osp.exists(a.refs):
        a.fstep = framecount // len(a.refs)
        refs = img_list(a.refs) if osp.isdir(a.refs) else [a.refs]
        print(' loading %d refs' % len(refs))
        cls_refs = [int(basename(fn)[0]) for fn in refs] # first char in ref name = class num
        img_refs = [Image.open(fname).convert('RGB') for fname in refs]
        img_refs = torch.cat([transform_ref(img).to(device).unsqueeze(0) for img in img_refs]) # [n,c,h,w]
        cls_refs = torch.tensor(cls_refs).to(device) # [b]  
        styles = nets_ema.style_encoder(img_refs, cls_refs).cpu().numpy() # [n,64]

    # make latents from list of classes
    else: 
        if 'all' in a.refs.lower():
            print(' testing %s domains' % a.refs)
            a.refs = '-'.join([str(x) for x in list(range(a.num_domains))])
            
        cls_refs = [int(cls) for cls in a.refs.split('-')] # string to integers
        assert len(cls_refs) > 0, " No refs found: %s" % a.refs
        assert all([ref < a.num_domains for ref in cls_refs]), " Refs out of domains: %d" % a.num_domains

        if a.fstep is None:
            a.fstep = framecount // len(cls_refs)

        stepcount = framecount // a.fstep
        if a.rand is True:
            cls_refs = [random.choice(cls_refs) for _ in range(stepcount)]
        if len(cls_refs) < stepcount: # repeat blocks 
            cls_refs = cls_refs * math.ceil(stepcount / len(cls_refs))
        cls_refs = cls_refs[:stepcount] # drop extra refs
        styles = gen_styles(cls_refs) # [n,64]

    # smooth animation
    if a.fstep > 1:
        s_shape = (1, a.style_dim)
        styles = latent_anima(s_shape, framecount, a.fstep, key_latents=styles, cubic=True, seed=a.seed, verbose=True) # [n,64]

    styles = torch.tensor(styles.astype(np.float32)).to(device).unsqueeze(1) # [n,1,64]

    framecount = styles.shape[0]

    pbar = ProgressBar(framecount)
    for i in range(framecount):

        if src_count > 1: # source = sequence
            img_src = Image.open(srcs[i % src_count]).convert('RGB')
            if out_size is None: out_size = img_src.size[::-1]

            img_src = transform_norm(T.ToTensor()(img_src).to(device)).unsqueeze(0) # slightly faster
            # img_src = transform_src(img_src).to(device).unsqueeze(0)

            if list(img_src.shape[-2:]) != out_size:
                img_src = F.interpolate(img_src, size=out_size, mode='bicubic', align_corners=True)

        style = styles[i] # [64]
        img_out = nets_ema.generator(img_src, style)
        if img_out.shape[-2:][::-1] != out_size:
            img_out = F.interpolate(img_out, size=out_size, mode='bicubic', align_corners=True)

        img_out = utils.post_np(img_out)
        checkout(img_out[0], i)
        pbar.upd()


if __name__ == '__main__':
    main()
