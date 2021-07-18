""" StarGAN v2
originally (c) 2020-present NAVER Corp.
http://creativecommons.org/licenses/by-nc/4.0/ 
"""

import os
import torch

class CheckpointIO(object):
    def __init__(self, fdir, model_name, model_postfix=None, device=None, parallel=False, **kwargs):
        if fdir=='': print(' !! Wrong model path, not found !!'); exit(1)
        os.makedirs(fdir, exist_ok=True)
        self.fdir = fdir
        self.model_name = model_name
        self.model_postfix = '' if model_postfix is None else model_postfix
        self.module_dict = kwargs
        self.parallel = parallel
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

    def register(self, **kwargs):
        self.module_dict.update(kwargs)

    def save(self, id): # id = integer epoch number
        fname = '%s-%03d%s.ckpt' % (self.model_name, id, self.model_postfix)
        fname = os.path.join(self.fdir, fname)
        # print(' Saving checkpoint %s...' % fname)
        outdict = {}
        for name, module in self.module_dict.items():
            try: # if self.parallel:
                outdict[name] = module.module.state_dict()
            except: # else:
                outdict[name] = module.state_dict()
        torch.save(outdict, fname)

    def load(self, id=None):
        try: # structured model name, id = integer epoch number
            fname = '%s-%03d%s.ckpt' % (self.model_name, id, self.model_postfix)
        except: # arbitrary model name (for inference)
            fname = '%s.ckpt' % (self.model_name)
        fname = os.path.join(self.fdir, fname)
        assert os.path.isfile(fname), fname + ' does not exist!'
        print(' Loading checkpoint %s...' % fname)
        module_dict = torch.load(fname, map_location=self.device)
        for name, module in self.module_dict.items():
            try: # if self.parallel:
                module.module.load_state_dict(module_dict[name])
            except: # else:
                module.load_state_dict(module_dict[name])

