"""
https://github.com/arfafax/StyleGAN2_experiments
Stochastic Weight Averaging: https://arxiv.org/abs/1803.05407
See: https://github.com/kristpapadopoulos/keras-stochastic-weight-averaging
"""
import os
import argparse

import torch

from utilgan import file_list, basename
try: # progress bar for notebooks 
    get_ipython().__class__.__name__
    from progress_bar import ProgressIPy as ProgressBar
except: # normal console
    from progress_bar import ProgressBar

parser = argparse.ArgumentParser(description='Perform stochastic weight averaging', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-i', '--in_dir', default='./', help='Directory with network checkpoints for weight averaging')
parser.add_argument('-o', '--out_dir', default='./', help='The averaged model to output')
parser.add_argument('--count', default=None, help='Average the last n checkpoints', type=int)
args = parser.parse_args()

def moving_average(model_main, model_new, epoch):
    scale_new_data = 1. / (epoch + 1)
    gen1 = model_main['generator']
    gen2 = model_new ['generator']
    for param_main, param_new in zip(gen1, gen2):
        if param_main == param_new:
            gen1[param_main] = torch.lerp(gen1[param_main], gen2[param_new], scale_new_data)
        else:
            print(' Inconsistent params:', param_main, param_new); exit(1)
    model_main['generator'] = gen1
    return model_main

def main():
    models = file_list(args.in_dir, 'ckpt')
    # print(len(models))
    models = [m for m in models if (not '_nets' in m) and (not '_optim' in m)]
    if args.count is not None:
        if (len(models) > args.count):
            models = models[-args.count:]
    
    mixname = basename(args.in_dir) if basename(args.in_dir) != '' else 'mix'
    model_name = '-'.join(basename(models[0]).split('-')[:3]) + '-%s.ckpt' % mixname
    file_out = os.path.join(args.out_dir, model_name)
    
    model_main = torch.load(models[0], map_location=torch.device('cpu'))
    
    pbar = ProgressBar(len(models))
    for i in range(len(models)):
        model_main = moving_average(model_main, torch.load(models[i], map_location=torch.device('cpu')), i)
        pbar.upd()
    
    torch.save(model_main, file_out)
    print(' Averaged model ::', file_out)

main()
