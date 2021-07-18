""" StarGAN v2
Copyright (c) 2020-present NAVER Corp.
http://creativecommons.org/licenses/by-nc/4.0/ 
"""

import os
from os.path import join as ospj
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.utils as vutils

def print_network(network, name):
    num_params = 0
    for p in network.parameters():
        num_params += p.numel()
    # print(network)
    print("Number of parameters of %s: %i" % (name, num_params))

def he_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

def denormalize(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def save_image(x, ncol, filename):
    x = denormalize(x)
    vutils.save_image(x.cpu(), filename, nrow=ncol, padding=0)

@torch.no_grad()
def translate_and_reconstruct(nets, args, x_src, y_src, x_ref, y_ref, filename):
    N, C, H, W = x_src.size()
    s_ref = nets.style_encoder(x_ref, y_ref)
    x_fake = nets.generator(x_src, s_ref)
    s_src = nets.style_encoder(x_src, y_src)
    x_rec = nets.generator(x_fake, s_src)
    x_concat = [x_src, x_ref, x_fake, x_rec] # [ [b,c,h,w] ..]
    x_concat = torch.cat(x_concat, dim=3)
    save_image(x_concat, 1, filename)
    del x_concat

@torch.no_grad()
def translate_using_latent(nets, args, x_src, y_trg_list, z_trg_list, psi, filename):
    N, C, H, W = x_src.size()
    latent_dim = z_trg_list[0].size(1)
    x_concat = []

    for i, y_trg in enumerate(y_trg_list):
        z_many = torch.randn(10000, latent_dim).to(x_src.device)
        y_many = torch.LongTensor(10000).to(x_src.device).fill_(y_trg[0])
        s_many = nets.mapping_network(z_many, y_many)
        s_avg = torch.mean(s_many, dim=0, keepdim=True)
        s_avg = s_avg.repeat(N, 1)

        for z_trg in z_trg_list:
            s_trg = nets.mapping_network(z_trg, y_trg)
            s_trg = torch.lerp(s_avg, s_trg, psi)
            x_fake = nets.generator(x_src, s_trg)
            x_concat += [x_fake]

    # x_concat = torch.cat(x_concat, dim=0)
    # save_image(x_concat, N, filename)
    x_concat = torch.cat(x_concat, dim=3)
    save_image(x_concat, 1, filename)

@torch.no_grad()
def translate_using_reference(nets, args, x_src, x_ref, y_ref, filename):
    N, C, H, W = x_src.size()
    wb = torch.ones(1, C, H, W).to(x_src.device)
    x_src_with_wb = torch.cat([wb, x_src], dim=0)

    s_ref = nets.style_encoder(x_ref, y_ref)
    s_ref_list = s_ref.unsqueeze(1).repeat(1, N, 1)
    x_concat = [x_src_with_wb]
    for i, s_ref in enumerate(s_ref_list):
        x_fake = nets.generator(x_src, s_ref)
        x_fake_with_ref = torch.cat([x_ref[i:i+1], x_fake], dim=0)
        x_concat += [x_fake_with_ref]

    x_concat = torch.cat(x_concat, dim=0)
    save_image(x_concat, N+1, filename)
    del x_concat

@torch.no_grad()
def debug_image(nets, args, inputs, step):
    x_src, y_src = inputs.x_src, inputs.y_src
    x_ref, y_ref = inputs.x_ref, inputs.y_ref

    device = inputs.x_src.device
    N = inputs.x_src.size(0)

    # translate and reconstruct (reference-guided)
    filename = ospj(args.sample_dir, '%03d_cycle_consistency.jpg' % (step//1000))
    translate_and_reconstruct(nets, args, x_src, y_src, x_ref, y_ref, filename)

    # latent-guided image synthesis
    y_trg_list = [torch.tensor(y).repeat(N).to(device) for y in range(args.num_domains)] # min(args.num_domains, 5)
    z_trg_list = torch.randn(args.sample_per_domain, 1, args.latent_dim).repeat(1, N, 1).to(device)
    psi = 1.
    filename = ospj(args.sample_dir, '%03d_latent.jpg' % (step//1000))
    translate_using_latent(nets, args, x_src, y_trg_list, z_trg_list, psi, filename)

    # reference-guided image synthesis
    filename = ospj(args.sample_dir, '%03d_reference.jpg' % (step//1000))
    translate_using_reference(nets, args, x_src, x_ref, y_ref, filename)

def post_np(images):
    images = torch.clamp(images * 0.5 + 0.5, 0, 1)
    images = images.cpu().numpy().transpose(0,2,3,1)
    images = (images * 255).astype(np.uint8)
    return images

