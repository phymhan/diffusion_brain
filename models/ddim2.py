""" SAMPLING ONLY.
Rewrite according to DiffusionCLIP, better for inversion.
"""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, \
    extract_into_tensor

import pdb
st = pdb.set_trace

class DDIM2Sampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self):
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

    @torch.no_grad()
    def sample(self,
        S,
        batch_size,
        shape,
        conditioning=None,
        callback=None,
        normals_sequence=None,
        img_callback=None,
        quantize_x0=False,
        eta=0.,
        mask=None,
        x0=None,
        temperature=1.,
        noise_dropout=0.,
        score_corrector=None,
        corrector_kwargs=None,
        verbose=True,
        x_T=None,
        log_every_t=100,
        unconditional_guidance_scale=1.,
        unconditional_conditioning=None,
        model_based_CFG=False,
        **kwargs
    ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                try:
                    cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                    if cbs != batch_size:
                        print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
                except:
                    print(f"conditioning[list(conditioning.keys())[0]] is not a tensor") 
            else:
                try:
                    if conditioning.shape[0] != batch_size:
                        print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")
                except:
                    print(f"conditioning is not a tensor")

        device = self.model.betas.device
        C, H, W = shape
        size = (batch_size, C, H, W)

        model_kwargs = {
            'conditioning': conditioning,
            'unconditional_conditioning': unconditional_conditioning,
            'unconditional_guidance_scale': unconditional_guidance_scale,
            'score_corrector': score_corrector,
            'corrector_kwargs': corrector_kwargs,
            'model_based_CFG': model_based_CFG,
            'range_t': self.model.extra_config['range_t'],
        }

        # NOTE: make time schedule
        self.make_schedule()
        ddim_num_steps = S
        # seq = (np.linspace(0, 1, ddim_num_steps) * self.ddpm_num_timesteps).astype(int)
        seq = list(range(0, self.ddpm_num_timesteps, self.ddpm_num_timesteps // ddim_num_steps))
        seq_next = [-1] + list(seq[:-1])

        if x_T is None:
            xt = torch.randn(size, device=device)
        else:
            xt = x_T
        
        intermediates = {'x_inter': [xt], 'pred_x0': [xt]}
        total_steps = len(seq) - 1
        iterator = tqdm(zip(reversed((seq[1:])), reversed((seq_next[1:]))), desc='DDIM Sampler', total=total_steps)

        for it, (i, j) in enumerate(iterator):
            t, t_next = i, j
            xt, xt_0 = self.denoising_step(
                xt, t=t, t_next=t_next,
                model=self.model,
                alphas_cumprod=self.alphas_cumprod,
                eta=eta,
                model_kwargs=model_kwargs,
            )
            if it % log_every_t == 0 or it == total_steps - 1:
                intermediates['x_inter'].append(xt)
                intermediates['pred_x0'].append(xt_0)

        return xt, intermediates
    
    def denoising_step(self,
        xt, t, t_next,
        model,
        alphas_cumprod,
        eta=0.0,
        model_kwargs=None,
    ):
        device = self.model.betas.device
        b = xt.shape[0]
        if model_kwargs is None:
            model_kwargs = {}

        # Compute noise and variance
        # if type(models) != list:
        et = self.apply_model(model, xt, t=t, **model_kwargs)

        # Compute the next x
        at = torch.full((b, 1, 1, 1), alphas_cumprod[t], device=device)
        
        at_next = torch.full((b, 1, 1, 1), alphas_cumprod[t_next], device=device)
        xt_next = torch.zeros_like(xt)
        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()  # NOTE: pred_x0
        if eta == 0:
            xt_next = at_next.sqrt() * x0_t + (1 - at_next).sqrt() * et
        elif at > (at_next):
            print('Inversion process is only possible with eta = 0')
            raise ValueError
        else:
            c1 = eta * ((1 - at / (at_next)) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c2 * et + c1 * torch.randn_like(xt)

        return xt_next, x0_t

    @torch.no_grad()
    def apply_model(self,
        model, x, t, score_corrector=None, corrector_kwargs=None,
        conditioning=None, unconditional_conditioning=None, unconditional_guidance_scale=1.,
        model_based_CFG=False, range_t=-1,
    ):
        device = self.model.betas.device
        step = t
        b = x.shape[0]
        c = conditioning
        t = torch.full((b,), t, device=device, dtype=torch.long)

        if not model_based_CFG:
            if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                e_t = model.apply_model(x, t, c)
            else:
                if unconditional_guidance_scale == 0.:
                    e_t = model.apply_model(x, t, unconditional_conditioning)
                else:
                    e_t_uncond = model.apply_model(x, t, unconditional_conditioning)
                    e_t = model.apply_model(x, t, c)
                    e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
        else:  # NOTE: perform model-based CFG
            if range_t > 0 and step >= range_t:
                # NOTE: ========== assume the first is single-model score ==========
                betas = model.extra_config['cond_betas1']
                c_mix = c
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t] * 2)
                # single score and uncond score
                c1 = torch.cat([c_mix[:1,...]] * b)
                c_in = torch.cat([unconditional_conditioning, c1])
                e_t_uncond, e_t1 = model.apply_model(x_in, t_in, c_in).chunk(2)
                e_t1 = e_t1 * betas[0]
                # other cond scores
                e_t2s = 0.
                for j in range(1, c_mix.shape[0]):
                    c2 = torch.cat([c_mix[j:j+1,...]] * b)
                    e_t2 = model.apply_model(x, t, c2)
                    e_t2s += e_t2 * betas[j]
                e_t = e_t_uncond + unconditional_guidance_scale * (e_t1 + e_t2s - e_t_uncond)
            else:
                betas = model.extra_config['cond_betas2']
                c_mix = c
                e_t_uncond = model.apply_model(x, t, unconditional_conditioning)
                # other cond scores
                if sum(betas[1:]) > 0:
                    e_t2s = 0.
                    for j in range(1, c_mix.shape[0]):
                        c2 = torch.cat([c_mix[j:j+1,...]] * b)
                        e_t2 = model.apply_model(x, t, c2)
                        e_t2s += e_t2 * betas[j]
                    e_t = e_t_uncond + unconditional_guidance_scale * (e_t2s - e_t_uncond * sum(betas[1:]))
                else:
                    e_t = e_t_uncond

        if score_corrector is not None:
            assert model.parameterization == "eps"
            e_t = score_corrector.modify_score(model, e_t, x, t, c, **corrector_kwargs)
        return e_t
    
    @torch.no_grad()
    def ddim_inversion(self,
        S,
        batch_size,
        shape,
        conditioning=None,
        eta=0.,
        mask=None,
        x0=None,
        score_corrector=None,
        corrector_kwargs=None,
        verbose=True,
        x_T=None,
        log_every_t=100,
        unconditional_guidance_scale=1.,
        unconditional_conditioning=None,
        **kwargs
    ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                try:
                    cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                    if cbs != batch_size:
                        print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
                except:
                    print(f"conditioning[list(conditioning.keys())[0]] is not a tensor") 
            else:
                try:
                    if conditioning.shape[0] != batch_size:
                        print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")
                except:
                    print(f"conditioning is not a tensor")

        model_kwargs = {
            'conditioning': conditioning,
            'unconditional_conditioning': unconditional_conditioning,
            'unconditional_guidance_scale': unconditional_guidance_scale,
            'score_corrector': score_corrector,
            'corrector_kwargs': corrector_kwargs,
        }

        # NOTE: make time schedule
        self.make_schedule()
        ddim_num_steps = S
        # seq = (np.linspace(0, 1, ddim_num_steps) * self.ddpm_num_timesteps).astype(int)
        seq = list(range(0, self.ddpm_num_timesteps, self.ddpm_num_timesteps // ddim_num_steps))
        seq_next = [-1] + list(seq[:-1])

        assert x0 is not None
        xt = x0
        
        intermediates = {'x_inter': [xt], 'pred_x0': [xt]}
        total_steps = len(seq) - 1
        iterator = tqdm(zip(seq_next[1:], seq[1:]), desc='DDIM Inversion', total=total_steps)

        for it, (i, j) in enumerate(iterator):
            t, t_next = i, j

            xt, xt_0 = self.denoising_step(
                xt, t=t, t_next=t_next,
                model=self.model,
                alphas_cumprod=self.alphas_cumprod,
                eta=eta,
                model_kwargs=model_kwargs,
            )
            if it % log_every_t == 0 or it == total_steps - 1:
                intermediates['x_inter'].append(xt)
                intermediates['pred_x0'].append(xt_0)

        return xt, intermediates
