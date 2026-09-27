"""A tiny stand-in for Tayeb's tFUS-FNO, behind the same interface.

Delete this module once the pinned ``tfus_fno`` package exists; see
babelbrain-docs/specs/starfish-work-items.md, SF-04. It is a small 3D Fourier
neural operator with the paper's shape, lift, Fourier blocks with four corner
weight blocks and a pointwise bypass, projection, scaled down so that three
FL rounds on 16 x 16 x 32 synthetic samples run on a CPU in seconds.

Interface, as agreed for ``tfus_fno``:

- ``build_model(bucket_hz, **cfg) -> torch.nn.Module``
- ``inputs(sample) -> (x, y, aux)``, aux holding ``sos``, ``attenuation``, ``brain_mask``
- ``loss(pred, y, aux, stage_weights) -> (total, parts)``
- ``ARCH_HASH``, which changes whenever the architecture changes

Normalisation here is the stand-in's own choice, not the paper's: CT in
thousands of HU and both fields divided by the in-water peak. The real
package owns its normalisation; see the CONFIRM items in the sample contract.
"""

import hashlib
import math

import torch
from torch import nn

NAME = 'standin'
ARCH = {'version': 1, 'in_channels': 6, 'out_channels': 2, 'width': 8, 'modes': (4, 4, 4),
        'layers': 2}
ARCH_HASH = hashlib.sha256(
    repr(sorted(ARCH.items())).encode()).hexdigest()[:16]


class SpectralConv3d(nn.Module):
    """Keeps the lowest ``modes`` along each axis, with four corner weight blocks.

    Weights are real tensors with a trailing axis of 2 for real and imaginary
    parts, so every parameter is a plain float tensor.
    """

    def __init__(self, width, modes):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (width * width)
        self.weights = nn.ParameterList([
            nn.Parameter(scale * torch.randn(width, width, *modes, 2)) for _ in range(4)])

    def forward(self, x):
        m1, m2, m3 = self.modes
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))
            out = torch.zeros_like(x_ft)
            corners = [(slice(None, m1), slice(None, m2)), (slice(-m1, None), slice(None, m2)),
                       (slice(None, m1), slice(-m2, None)), (slice(-m1, None), slice(-m2, None))]
            for (s1, s2), w in zip(corners, self.weights):
                out[:, :, s1, s2, :m3] = torch.einsum(
                    'bixyz,ioxyz->boxyz', x_ft[:, :, s1, s2, :m3], torch.view_as_complex(w))
            return torch.fft.irfftn(out, s=x.shape[-3:], dim=(-3, -2, -1))


class FourierBlock(nn.Module):

    def __init__(self, width, modes):
        super().__init__()
        self.spectral = SpectralConv3d(width, modes)
        self.bypass = nn.Conv3d(width, width, 1)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.spectral(x) + self.bypass(x))


class StandinFNO(nn.Module):

    def __init__(self, width, modes, layers, grad_checkpointing=False):
        super().__init__()
        self.lift = nn.Conv3d(ARCH['in_channels'], width, 1)
        self.blocks = nn.ModuleList(
            [FourierBlock(width, modes) for _ in range(layers)])
        self.project = nn.Sequential(nn.Conv3d(width, width, 1), nn.GELU(),
                                     nn.Conv3d(width, ARCH['out_channels'], 1))
        self.grad_checkpointing = grad_checkpointing

    def forward(self, x):
        x = self.lift(x)
        for block in self.blocks:
            if self.grad_checkpointing and x.requires_grad:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.project(x)


def build_model(bucket_hz, grad_checkpointing=False, **cfg):
    """The stand-in network. ``bucket_hz`` is accepted for interface parity."""
    return StandinFNO(ARCH['width'], ARCH['modes'], ARCH['layers'], grad_checkpointing)


def _batched(t):
    return t if t.dim() >= 5 else t.unsqueeze(0)


def coordinate_grid(shape, like):
    """Three channels of normalised x, y and z coordinates in [0, 1]."""
    axes = [torch.linspace(0, 1, n, device=like.device,
                           dtype=like.dtype) for n in shape]
    return torch.stack(torch.meshgrid(*axes, indexing='ij'))


def inputs(sample):
    """Model input, label and auxiliary maps from one dataset item or a batch of them."""
    ct = sample['ct_hu'].float()
    if ct.dim() == 3:
        ct = ct.unsqueeze(0)
    water = _batched(sample['water_field'].float())
    skull = _batched(sample['skull_field'].float())
    batch = water.shape[0]
    scale = water.flatten(1).abs().amax(
        dim=1).clamp_min(1e-12).view(batch, 1, 1, 1, 1)
    grid = coordinate_grid(
        water.shape[-3:], water).unsqueeze(0).expand(batch, -1, -1, -1, -1)
    x = torch.cat([ct.unsqueeze(1) / 1000.0, water / scale, grid], dim=1)
    y = skull / scale
    aux = {}
    for name in ('sos', 'attenuation', 'brain_mask'):
        a = sample[name].float()
        aux[name] = a.unsqueeze(0) if a.dim() == 3 else a
    freq = sample.get('frequency_hz', 250000.0)
    spacing = sample.get('spacing_mm', 0.49)
    aux['frequency_hz'] = torch.as_tensor(
        freq, dtype=torch.float32).reshape(-1)[:1]
    aux['spacing_m'] = torch.as_tensor(
        spacing, dtype=torch.float32).reshape(-1)[:1] / 1000.0
    return x, y, aux


def relative_l2(pred, y):
    """Mean over the batch of ||pred - y|| / ||y|| on the two-channel complex field."""
    diff = (pred - y).flatten(1).norm(dim=1)
    ref = y.flatten(1).norm(dim=1).clamp_min(1e-12)
    return (diff / ref).mean()


def _gradients(field):
    return [torch.diff(field, dim=d) for d in (-3, -2, -1)]


def helmholtz_residual(pred, aux):
    """Relative residual of laplacian(p) + k^2 p inside the brain mask, attenuation ignored."""
    h = aux['spacing_m'].to(pred)
    k = 2 * math.pi * aux['frequency_hz'].to(pred) / aux['sos'].clamp_min(1.0)
    p = pred
    inner = p[..., 1:-1, 1:-1, 1:-1]
    # Seven-point Laplacian, on the grid without its outer layer
    lap = (p[..., 2:, 1:-1, 1:-1] + p[..., :-2, 1:-1, 1:-1]
           + p[..., 1:-1, 2:, 1:-1] + p[..., 1:-1, :-2, 1:-1]
           + p[..., 1:-1, 1:-1, 2:] + p[..., 1:-1, 1:-1, :-2] - 6 * inner) / (h * h)
    k2 = (k[..., 1:-1, 1:-1, 1:-1] ** 2).unsqueeze(1)
    mask = aux['brain_mask'][..., 1:-1, 1:-1, 1:-1].unsqueeze(1)
    residual = (lap + k2 * inner) * mask
    ref = (k2 * inner * mask).flatten(1).norm(dim=1).clamp_min(1e-12)
    return (residual.flatten(1).norm(dim=1) / ref).mean()


def loss(pred, y, aux, stage_weights):
    """Relative l2, plus an H1 term and a Helmholtz residual weighted by ``stage_weights``."""
    parts = {'rel_l2': relative_l2(pred, y)}
    total = parts['rel_l2']
    h1_weight = float(stage_weights.get('h1_weight', 0.0))
    pde_weight = float(stage_weights.get('pde_weight', 0.0))
    if h1_weight:
        gp, gy = _gradients(pred), _gradients(y)
        parts['h1'] = sum(relative_l2(a, b) for a, b in zip(gp, gy)) / 3
        total = total + h1_weight * parts['h1']
    if pde_weight:
        parts['pde'] = helmholtz_residual(pred, aux)
        total = total + pde_weight * parts['pde']
    return total, {name: float(value.detach()) for name, value in parts.items()}
