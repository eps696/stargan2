""" StarGAN v2
Copyright (c) 2020-present NAVER Corp. [edited]
http://creativecommons.org/licenses/by-nc/4.0/ 
"""

import os
import sys
from munch import Munch

import torch
import torch.nn as nn
import torch.nn.functional as F
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from torch.utils.tensorboard import SummaryWriter

from core.model import build_model
from core.checkpoint import CheckpointIO
from core.data_loader import InputFetcher
import core.utils as utils

sys.path.append(os.path.dirname(os.path.dirname(__file__))) # upper dir
try: # progress bar for notebooks 
    get_ipython().__class__.__name__
    from progress_bar import ProgressIPy as ProgressBar
except: # normal console
    from progress_bar import ProgressBar

class Solver(nn.Module):
    def __init__(self, args, device=None, parallel=False):
        super().__init__()
        self.args = args
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        # print(self.device)

        self.nets, self.nets_ema = build_model(args)
        # below setattrs are to make networks be children of Solver, e.g., for self.to(self.device)
        for name, module in self.nets.items():
            # utils.print_network(module, name)
            setattr(self, name, module)
        for name, module in self.nets_ema.items():
            setattr(self, name + '_ema', module)

        if args.mode == 'train':
            self.optims = Munch()
            for net in self.nets.keys():
                self.optims[net] = torch.optim.Adam(
                    params=self.nets[net].parameters(),
                    lr=args.f_lr if net == 'mapping_network' else args.lr,
                    betas=[args.beta1, args.beta2],
                    weight_decay=args.weight_decay)

            model_name = '%s-%d-%d' % (os.path.basename(args.model_dir), args.img_size, args.num_domains)
            self.ckptios = [
                CheckpointIO(args.model_dir, model_name, '',        self.device, parallel=parallel, **self.nets_ema),
                CheckpointIO(args.model_dir, model_name, '_nets',   self.device, parallel=parallel, **self.nets),
                CheckpointIO(args.model_dir, model_name, '_optims', self.device, **self.optims)]
        else:
            self.ckptios = [CheckpointIO(args.model_dir, args.model, '', **self.nets_ema)]

        self.to(self.device)
        for name, network in self.named_children():
            if 'ema' not in name:
                network.apply(utils.he_init)

    def _save_checkpoints(self, num):
        for ckptio in self.ckptios:
            ckptio.save(num)

    def _load_checkpoints(self, num):
        for ckptio in self.ckptios:
            ckptio.load(num)

    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()

    def train(self, loaders):
        args = self.args
        nets = self.nets
        nets_ema = self.nets_ema
        optims = self.optims

        fetcher = InputFetcher(loaders.src, loaders.ref, args.latent_dim, 'train', device=self.device)
        fetcher_val = InputFetcher(loaders.val, None, args.latent_dim, 'val', device=self.device)
        inputs_val = next(fetcher_val)

        # resume training if necessary
        if args.resume > 0:
            self._load_checkpoints(args.resume)

        # remember the initial value of ds weight
        initial_lambda_ds = args.lambda_ds

        print('Start training...')
        tblog = SummaryWriter(args.model_dir)
        pbar = ProgressBar(args.total_iters, args.resume*1000)
        for i in range(args.resume*1000, args.total_iters):
            # fetch images and labels
            inputs = next(fetcher)

            # train the discriminator
            d_loss, d_losses_latent = compute_d_loss(nets, args, inputs.x_src, inputs.y_src, inputs.y_ref, z_trg=inputs.z_trg)
            self._reset_grad()
            d_loss.backward()
            optims.discriminator.step()

            d_loss, d_losses_ref = compute_d_loss(nets, args, inputs.x_src, inputs.y_src, inputs.y_ref, x_ref=inputs.x_ref)
            self._reset_grad()
            d_loss.backward()
            optims.discriminator.step()
            
            # train the generator
            g_loss, g_losses_latent = compute_g_loss(nets, args, inputs.x_src, inputs.y_src, inputs.y_ref, z_trgs=[inputs.z_trg, inputs.z_trg2])
            self._reset_grad()
            g_loss.backward()
            optims.generator.step()
            optims.mapping_network.step()
            optims.style_encoder.step()
            
            g_loss, g_losses_ref = compute_g_loss(nets, args, inputs.x_src, inputs.y_src, inputs.y_ref, x_refs=[inputs.x_ref, inputs.x_ref2])
            self._reset_grad()

            g_loss.backward()
            optims.generator.step()

            # compute moving average of network parameters
            moving_average(nets.generator, nets_ema.generator, beta=0.999)
            moving_average(nets.mapping_network, nets_ema.mapping_network, beta=0.999)
            moving_average(nets.style_encoder, nets_ema.style_encoder, beta=0.999)

            # decay weight for diversity sensitive loss
            if args.lambda_ds > 0:
                args.lambda_ds -= (initial_lambda_ds / args.ds_iter)

            # print out log info
            if (i+1) % args.log_every == 0:
                for loss, prefix in zip([d_losses_latent, d_losses_ref, g_losses_latent, g_losses_ref], ['D/latent_', 'D/ref_', 'G/latent_', 'G/ref_']):
                    for key, value in loss.items():
                        tblog.add_scalar(prefix + key, value, i)
                tblog.add_scalar('G/lambda_ds', args.lambda_ds, i)
            if i == args.resume*1000: pbar.reset(args.resume*1000) # drop first long cycle
            else: pbar.upd()

            # generate images for debugging
            if (i+1) % args.sample_every == 0:
                os.makedirs(args.sample_dir, exist_ok=True)
                utils.debug_image(nets_ema, args, inputs=inputs_val, step=i+1)
                self.ckptios[0].save((i+1)//1000) # save generator

            # save model checkpoints
            if (i+1) % args.save_every == 0:
                self._save_checkpoints((i+1)//1000)

def compute_d_loss(nets, args, x_real, y_org, y_trg, z_trg=None, x_ref=None):
    assert (z_trg is None) != (x_ref is None)
    # with real images
    x_real.requires_grad_()

    out = nets.discriminator(x_real, y_org)
    loss_real = adv_loss(out, 1)
    loss_reg = r1_reg(out, x_real)

    # with fake images
    with torch.no_grad():
        if z_trg is not None:
            s_trg = nets.mapping_network(z_trg, y_trg)
        else:  # x_ref is not None
            s_trg = nets.style_encoder(x_ref, y_trg) # (b,c,h,w), (b)

        x_fake = nets.generator(x_real, s_trg)

    out = nets.discriminator(x_fake, y_trg)
    loss_fake = adv_loss(out, 0)

    loss = loss_real + loss_fake + args.lambda_reg * loss_reg
    return loss, Munch(real=loss_real.item(), fake=loss_fake.item(), reg=loss_reg.item())

def compute_g_loss(nets, args, x_real, y_org, y_trg, z_trgs=None, x_refs=None):
    assert (z_trgs is None) != (x_refs is None)

    # adversarial loss
    if z_trgs is not None:
        s_trg = nets.mapping_network(z_trgs[0], y_trg)
    else:
        s_trg = nets.style_encoder(x_refs[0], y_trg)

    x_fake = nets.generator(x_real, s_trg)
    out = nets.discriminator(x_fake, y_trg)
    loss_adv = adv_loss(out, 1)

    # style reconstruction loss
    s_pred = nets.style_encoder(x_fake, y_trg)
    loss_sty = torch.mean(torch.abs(s_pred - s_trg))

    # diversity sensitive loss
    if z_trgs is not None:
        s_trg2 = nets.mapping_network(z_trgs[1], y_trg)
    else:
        s_trg2 = nets.style_encoder(x_refs[1], y_trg)

    x_fake2 = nets.generator(x_real, s_trg2)
    x_fake2 = x_fake2.detach()
    loss_ds = torch.mean(torch.abs(x_fake - x_fake2))

    # cycle-consistency loss
    s_org = nets.style_encoder(x_real, y_org)
    
    del x_fake2, y_org, y_trg, s_trg2, s_trg, s_pred, out 
    # torch.cuda.empty_cache()
    
    x_rec = nets.generator(x_fake, s_org)
    loss_cyc = torch.mean(torch.abs(x_rec - x_real))

    loss = loss_adv + args.lambda_sty * loss_sty - args.lambda_ds * loss_ds + args.lambda_cyc * loss_cyc
    return loss, Munch(adv=loss_adv.item(), sty=loss_sty.item(), ds=loss_ds.item(), cyc=loss_cyc.item())

def moving_average(model, model_test, beta=0.999):
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = torch.lerp(param.data, param_test.data, beta)

def adv_loss(logits, target):
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss

def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(outputs=d_out.sum(), inputs=x_in, create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg

