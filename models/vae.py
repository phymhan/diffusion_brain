"""
AUTOENCODER WITH ARCHTECTURE FROM VERSION 2
"""
from typing import Tuple, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.vae import AutoencoderKLOutput, DiagonalGaussianDistribution, DecoderOutput


@torch.jit.script
def swish(x):
    return x * torch.sigmoid(x)


def Normalize(in_channels):
    return nn.GroupNorm(
        num_groups=32,
        num_channels=in_channels,
        eps=1e-6,
        affine=True
    )


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=2,
            padding=0
        )

    def forward(self, x):
        pad = (0, 1, 0, 1, 0, 1)
        x = nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.norm2 = Normalize(out_channels)
        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0
            )

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class Encoder(nn.Module):
    def __init__(
            self,
            in_channels: int,
            n_channels: int,
            z_channels: int,
            ch_mult: Tuple[int],
            num_res_blocks: int,
            resolution: Tuple[int],
            attn_resolutions: Tuple[int],
            **ignorekwargs,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.n_channels = n_channels
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.attn_resolutions = attn_resolutions

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)

        blocks = []
        # initial convolution
        blocks.append(
            nn.Conv3d(
                in_channels,
                n_channels,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )

        # residual and downsampling blocks, with attention on smaller res (16x16)
        for i in range(self.num_resolutions):
            block_in_ch = n_channels * in_ch_mult[i]
            block_out_ch = n_channels * ch_mult[i]
            for _ in range(self.num_res_blocks):
                blocks.append(ResBlock(block_in_ch, block_out_ch))
                block_in_ch = block_out_ch

            if i != self.num_resolutions - 1:
                blocks.append(Downsample(block_in_ch))
                curr_res = tuple(ti // 2 for ti in curr_res)

        # normalise and convert to latent size
        blocks.append(Normalize(block_in_ch))
        blocks.append(
            nn.Conv3d(
                block_in_ch,
                z_channels,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class Decoder(nn.Module):
    def __init__(
            self,
            n_channels: int,
            z_channels: int,
            out_channels: int,
            ch_mult: Tuple[int],
            num_res_blocks: int,
            resolution: Tuple[int],
            attn_resolutions: Tuple[int],
            **ignorekwargs,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.z_channels = z_channels
        self.out_channels = out_channels
        self.ch_mult = ch_mult
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.attn_resolutions = attn_resolutions

        block_in_ch = n_channels * self.ch_mult[-1]
        curr_res = tuple(ti // 2 ** (self.num_resolutions - 1) for ti in resolution)

        blocks = []
        # initial conv
        blocks.append(
            nn.Conv3d(
                z_channels,
                block_in_ch,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )

        for i in reversed(range(self.num_resolutions)):
            block_out_ch = n_channels * self.ch_mult[i]

            for _ in range(self.num_res_blocks):
                blocks.append(ResBlock(block_in_ch, block_out_ch))
                block_in_ch = block_out_ch

            if i != 0:
                blocks.append(Upsample(block_in_ch))
                curr_res = tuple(ti * 2 for ti in curr_res)

        blocks.append(Normalize(block_in_ch))
        blocks.append(
            nn.Conv3d(
                block_in_ch,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class AutoencoderKL(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(
        self,
        embed_dim: int,
        in_channels: int = 3,
        out_channels: int = 3,
        n_channels: int = 64,
        z_channels: int = 3,
        ch_mult: List[int] = [1, 2, 2, 2],
        num_res_blocks: int = 2,
        resolution: List[int] = [160, 224, 160],
        attn_resolutions: List[int] = [],
        use_logvar: bool = False,
    ):
        super().__init__()
        hparams = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "n_channels": n_channels,
            "z_channels": z_channels,
            "ch_mult": ch_mult,
            "num_res_blocks": num_res_blocks,
            "resolution": resolution,
            "attn_resolutions": attn_resolutions,
        }
        self.encoder = Encoder(**hparams)
        self.decoder = Decoder(**hparams)
        self.quant_conv_mu = torch.nn.Conv3d(hparams["z_channels"], embed_dim, 1)
        self.quant_conv_log_sigma = torch.nn.Conv3d(hparams["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv3d(embed_dim, hparams["z_channels"], 1)
        self.embed_dim = embed_dim
        self.use_slicing = False  # NOTE: hardcoded
        self.use_logvar = use_logvar

    def encode(self, x: torch.FloatTensor, return_dict: bool = True, deterministic: bool = True) -> AutoencoderKLOutput:
        h = self.encoder(x)
        mu = self.quant_conv_mu(h)
        logvar = self.quant_conv_log_sigma(h)
        if not self.use_logvar:
            logvar = 2 * logvar  # NOTE: `logvar` is actually log_sigma
        moments = torch.cat([mu, logvar], dim=1)
        posterior = DiagonalGaussianDistribution(moments, deterministic=deterministic)
        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    def _decode(self, z: torch.FloatTensor, return_dict: bool = True) -> Union[DecoderOutput, torch.FloatTensor]:
        z = self.post_quant_conv(z)
        dec = self.decoder(z)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def decode_legacy(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def decode(self, z: torch.FloatTensor, return_dict: bool = True) -> Union[DecoderOutput, torch.FloatTensor]:
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice).sample for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z).sample

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)
    
    def forward(
        self,
        sample: torch.FloatTensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
        deterministic: bool = False,
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        r"""
        Args:
            sample (`torch.FloatTensor`): Input sample.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
        x = sample
        posterior = self.encode(x, deterministic=deterministic).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z).sample

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def reconstruct_ldm_outputs(self, z):
        x_hat = self.decode_legacy(z)
        return x_hat
