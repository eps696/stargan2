import os
import sys
import time
import math
import numpy as np
import collections
import scipy
from scipy.ndimage import gaussian_filter
from scipy.interpolate import CubicSpline as CubSpline

from imageio import imread

import torch
# import torch.nn.functional as F

def basename(file):
    return os.path.splitext(os.path.basename(file))[0]

def save_cfg(args, dir='./', file='config.txt'):
    if dir != '':
        os.makedirs(dir, exist_ok=True)
    try: args = vars(args)
    except: pass
    if file is None:
        print_dict(args)
    else:
        with open(os.path.join(dir, file), 'w') as cfg_file:
            print_dict(args, cfg_file)

def print_dict(dict, file=None, path="", indent=''):
    for k in sorted(dict.keys()):
        if isinstance(dict[k], collections.abc.Mapping):
            if file is None:
                print(indent + str(k))
            else:
                file.write(indent + str(k) + ' \n')
            path = k if path=="" else path + "->" + k
            print_dict(dict[k], file, path, indent + '   ')
        else:
            if file is None:
                print('%s%s: %s' % (indent, str(k), str(dict[k])))
            else:
                file.write('%s%s: %s \n' % (indent, str(k), str(dict[k])))

def dir_list(in_dir):
    dirs = [os.path.join(in_dir, x) for x in os.listdir(in_dir)]
    return sorted([f for f in dirs if os.path.isdir(f)])

def img_list(path, subdir=None):
    if subdir is True:
        files = [os.path.join(dp, f) for dp, dn, fn in os.walk(path) for f in fn]
    else:
        files = [os.path.join(path, f) for f in os.listdir(path)]
    files = [f for f in files if os.path.splitext(f.lower())[1][1:] in ['jpg', 'jpeg', 'png', 'ppm', 'tif']]
    files = [f for f in files if not '/__MACOSX/' in f.replace('\\', '/')] # workaround fix for macos phantom files
    return sorted([f for f in files if os.path.isfile(f)])

def file_list(path, ext=None, subdir=None):
    if subdir is True:
        files = [os.path.join(dp, f) for dp, dn, fn in os.walk(path) for f in fn]
    else:
        files = [os.path.join(path, f) for f in os.listdir(path)]
    if ext is not None: 
        if isinstance(ext, list):
            files = [f for f in files if os.path.splitext(f.lower())[1][1:] in ext]
        elif isinstance(ext, str):
            files = [f for f in files if f.endswith(ext)]
        else:
            print(' Unknown extension/type for file list!')
    return sorted([f for f in files if os.path.isfile(f)])

# = = = = = = = = = = = = = = = = = = = = = = = = = = = 

def get_z(shape, rnd, uniform=False):
    if uniform:
        return rnd.uniform(0., 1., shape)
    else:
        return rnd.randn(*shape) # *x unpacks tuple/list to sequence

def smoothstep(x, NN=1., xmin=0., xmax=1.):
    N = math.ceil(NN)
    x = np.clip((x - xmin) / (xmax - xmin), 0, 1)
    result = 0
    for n in range(0, N+1):
         result += scipy.special.comb(N+n, n) * scipy.special.comb(2*N+1, N-n) * (-x)**n
    result *= x**(N+1)
    if NN != N: result = (x + result) / 2
    return result

def lerp(z1, z2, num_steps, smooth=0.): 
    vectors = []
    xs = [step / (num_steps - 1) for step in range(num_steps)]
    if smooth > 0: xs = [smoothstep(x, smooth) for x in xs]
    for x in xs:
        interpol = z1 + (z2 - z1) * x
        vectors.append(interpol)
    return np.array(vectors)

# interpolate on hypersphere
def slerp(z1, z2, num_steps, smooth=0.):
    z1_norm = np.linalg.norm(z1)
    z2_norm = np.linalg.norm(z2)
    z2_normal = z2 * (z1_norm / z2_norm)
    vectors = []
    xs = [step / (num_steps - 1) for step in range(num_steps)]
    if smooth > 0: xs = [smoothstep(x, smooth) for x in xs]
    for x in xs:
        interplain = z1 + (z2 - z1) * x
        interp = z1 + (z2_normal - z1) * x
        interp_norm = np.linalg.norm(interp)
        interpol_normal = interplain * (z1_norm / interp_norm)
        # interpol_normal = interp * (z1_norm / interp_norm)
        vectors.append(interpol_normal)
    return np.array(vectors)

def cublerp(points, steps, fstep):
    keys = np.array([i*fstep for i in range(steps)] + [steps*fstep])
    points = np.concatenate((points, np.expand_dims(points[0], 0)))
    cspline = CubSpline(keys, points)
    return cspline(range(steps*fstep+1))

# = = = = = = = = = = = = = = = = = = = = = = = = = = = 
    
def latent_anima(shape, frames, transit, key_latents=None, smooth=0.5, uniform=False, cubic=False, gauss=False, start_lat=None, seed=None, verbose=True):
    if key_latents is None:
        transit = int(max(1, min(frames, transit)))
    steps = max(1, int(frames // transit))
    log = ' timeline: %d steps by %d' % (steps, transit)

    if seed is None:
        seed = np.random.seed(int((time.time()%1) * 9999))
    rnd = np.random.RandomState(seed)
    
    # make key points
    if key_latents is None:
        key_latents = np.array([get_z(shape, rnd, uniform) for i in range(steps)])
    if start_lat is not None:
        key_latents[0] = start_lat

    latents = np.expand_dims(key_latents[0], 0)
    
    # populate lerp between key points
    if transit == 1:
        latents = key_latents
    else:
        if cubic:
            latents = cublerp(key_latents, steps, transit)
            log += ', cubic'
        else:
            for i in range(steps):
                zA = key_latents[i]
                zB = key_latents[(i+1) % steps]
                if uniform is True:
                    interps_z = lerp(zA, zB, transit, smooth=smooth)
                else:
                    interps_z = slerp(zA, zB, transit, smooth=smooth)
                latents = np.concatenate((latents, interps_z))
    latents = np.array(latents)
    
    if gauss:
        lats_post = gaussian_filter(latents, [transit, 0, 0], mode="wrap")
        lats_post = (lats_post / np.linalg.norm(lats_post, axis=-1, keepdims=True)) * math.sqrt(np.prod(shape))
        log += ', gauss'
        latents = lats_post
        
    if verbose: print(log)
    if latents.shape[0] > frames: # extra frame
        latents = latents[1:]
    return latents
    
# = = = = = = = = = = = = = = = = = = = = = = = = = = = 
    
# Tiles an array around two points, allowing for pad lengths greater than the input length
# NB: if symm=True, every second tile is mirrored = messed up in GAN
# adapted from https://discuss.pytorch.org/t/symmetric-padding/19866/3
def tile_pad(xt, padding, symm=True):
    h, w = xt.shape[-2:]
    left, right, top, bottom = padding
 
    def tile(x, minx, maxx, symm=True):
        rng = maxx - minx
        if symm is True: # triangular reflection
            double_rng = 2*rng
            mod = np.fmod(x - minx, double_rng)
            normed_mod = np.where(mod < 0, mod+double_rng, mod)
            out = np.where(normed_mod >= rng, double_rng - normed_mod, normed_mod) + minx
        else: # repeating tiles
            mod = np.remainder(x - minx, rng)
            out = mod + minx
        return np.array(out, dtype=x.dtype)

    x_idx = np.arange(-left, w+right)
    y_idx = np.arange(-top, h+bottom)
    x_pad = tile(x_idx, -0.5, w-0.5, symm)
    y_pad = tile(y_idx, -0.5, h-0.5, symm)
    xx, yy = np.meshgrid(x_pad, y_pad)
    return xt[..., yy, xx]

def pad_up_to(x, size, type='centr'):
    sh = x.shape[2:][::-1]
    if list(x.shape[2:]) == list(size): return x
    padding = []
    for i, s in enumerate(size[::-1]):
        if 'side' in type.lower():
            padding = padding + [0, s-sh[i]]
        else: # centr
            p0 = (s-sh[i]) // 2
            p1 = s-sh[i] - p0
            padding = padding + [p0,p1]
    y = tile_pad(x, padding, symm = 'symm' in type.lower())
    return y
