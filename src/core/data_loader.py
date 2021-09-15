""" StarGAN v2
Copyright (c) 2020-present NAVER Corp.
http://creativecommons.org/licenses/by-nc/4.0/ 
"""

import os
import random

from munch import Munch
from PIL import Image
import numpy as np

import torch
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

def listdir(path):
    files = [os.path.join(path, f) for f in os.listdir(path)]
    files = [f for f in files if os.path.splitext(f.lower())[1][1:] in ['jpg', 'jpeg', 'png', 'ppm', 'tif']]
    files = [f for f in files if not '/__MACOSX/' in f.replace('\\', '/')] # workaround fix for macos phantom files
    return sorted([f for f in files if os.path.isfile(f)])

class ReferenceDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.samples, self.targets = self._make_dataset(root)
        self.transform = transform

    def _make_dataset(self, root):
        domains = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root,d))]
        fnames, fnames2, labels = [], [], []
        for idx, domain in enumerate(domains):
            class_dir = os.path.join(root, domain)
            cls_fnames = listdir(class_dir)
            fnames += cls_fnames
            fnames2 += random.sample(cls_fnames, len(cls_fnames))
            labels += [idx] * len(cls_fnames)
        return list(zip(fnames, fnames2)), labels

    def __getitem__(self, index):
        fname, fname2 = self.samples[index]
        label = self.targets[index]
        img = Image.open(fname).convert('RGB')
        img2 = Image.open(fname2).convert('RGB')
        if self.transform is not None:
            try:
                img = self.transform(img)
            except:
                print(fname); exit(11)
            try:
                img2 = self.transform(img2)
            except:
                print(fname2); exit(11)
        return img, img2, label

    def __len__(self):
        return len(self.targets)

# dataset loader for <root>/<class>/test subdirs
class TrainValDataset(data.Dataset):
    def __init__(self, root, transform=None, mode='train'):
        self.samples = self._make_dataset(root, mode)
        self.transform = transform
        self.targets = [s[1] for s in self.samples]

    def _make_dataset(self, root, mode):
        domains = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root,d))]
        if mode == 'train': print(' domains', domains)
        fnames, labels = [], []
        for idx, domain in enumerate(domains):
            class_dir = os.path.join(root, domain)
            if mode == 'test': 
                class_dir = os.path.join(class_dir, 'test')
            cls_fnames = listdir(class_dir)
            fnames += cls_fnames
            labels += [idx] * len(cls_fnames)
        return list(zip(fnames, labels))

    def __getitem__(self, index):
        fname, label = self.samples[index]
        img = Image.open(fname).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.samples)

def _make_balanced_sampler(labels):
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, len(weights))

def get_train_loader(root, which='src', img_size=256, batch_size=8, num_workers=4):
    transform = transforms.Compose([
        transforms.RandomCrop(img_size), 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5,.5,.5], std=[.5,.5,.5]),
    ])
    if   which == 'src':
        dataset = TrainValDataset(root, transform, mode='train')
    elif which == 'ref':
        dataset = ReferenceDataset(root, transform)
    elif which == 'val':
        dataset = TrainValDataset(root, transform, mode='test')
    else:
        raise NotImplementedError
    print(' training data %s: %d imgs' % (which, len(dataset)))

    sampler = _make_balanced_sampler(dataset.targets)
    return data.DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, drop_last=True)


class InputFetcher:
    def __init__(self, loader, loader_ref=None, latent_dim=16, mode='', device=None):
        self.loader = loader
        self.loader_ref = loader_ref
        self.latent_dim = latent_dim
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        # print(self.device)
        self.mode = mode

        self.iter = iter(self.loader)

    def _fetch_inputs(self):
        try:
            x, y = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            x, y = next(self.iter)
        return x, y

    def _fetch_refs(self):
        try:
            x, x2, y = next(self.iter_ref)
        except (AttributeError, StopIteration):
            self.iter_ref = iter(self.loader_ref)
            x, x2, y = next(self.iter_ref)
        return x, x2, y

    def __next__(self):
        x, y = self._fetch_inputs()
        if self.mode == 'train':
            x_ref, x_ref2, y_ref = self._fetch_refs()
            z_trg = torch.randn(x.size(0), self.latent_dim)
            z_trg2 = torch.randn(x.size(0), self.latent_dim)
            inputs = Munch(x_src=x, y_src=y, y_ref=y_ref,  x_ref=x_ref, x_ref2=x_ref2, z_trg=z_trg, z_trg2=z_trg2)
        elif self.mode == 'val':
            x_ref, y_ref = self._fetch_inputs()
            inputs = Munch(x_src=x, y_src=y,  x_ref=x_ref, y_ref=y_ref)
        else:
            raise NotImplementedError

        return Munch({k: v.to(self.device) for k, v in inputs.items()})

