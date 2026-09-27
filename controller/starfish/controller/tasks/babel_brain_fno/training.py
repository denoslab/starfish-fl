"""Local training for BabelBrainFno: device choice, one round of epochs, validation.

The model package, ``tfus_fno`` or the stand-in, supplies the network, the
input assembly and the loss. This module only runs the loop around them.
"""

import resource
import sys
import time


def pick_device(name='auto'):
    """``auto`` means CUDA, then Apple MPS, then CPU."""
    import torch
    if name and name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    mps = getattr(torch.backends, 'mps', None)
    if mps is not None and mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def stage_weights(curriculum, round_no):
    """Loss weights of the last curriculum stage that has started by ``round_no``.

    FL rounds replace the paper's epochs: each stage is
    ``{"from_round": r, "h1_weight": w1, "pde_weight": w2}``.
    """
    weights = {'h1_weight': 0.0, 'pde_weight': 0.0}
    for stage in sorted(curriculum or [], key=lambda s: int(s.get('from_round', 1))):
        if round_no >= int(stage.get('from_round', 1)):
            weights = {'h1_weight': float(stage.get('h1_weight', 0.0)),
                       'pde_weight': float(stage.get('pde_weight', 0.0))}
    return weights


def peak_memory_mb(device):
    """Peak GPU memory on CUDA, else the process's peak resident memory."""
    import torch
    if device.type == 'cuda':
        return torch.cuda.max_memory_allocated(device) / 2 ** 20
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is in bytes on macOS and in KiB on Linux
    return rss / 2 ** 20 if sys.platform == 'darwin' else rss / 1024


def _to(aux, device):
    return {k: v.to(device) if hasattr(v, 'to') else v for k, v in aux.items()}


def ssim3d(a, b, window=7):
    """Mean SSIM of two batches of 3D volumes, shape (B, X, Y, Z), with a uniform window."""
    import torch.nn.functional as F
    a, b = a.unsqueeze(1).float(), b.unsqueeze(1).float()
    pad = window // 2

    def mean(t):
        return F.avg_pool3d(t, window, stride=1, padding=pad, count_include_pad=False)

    data_range = (b.flatten(1).amax(1) - b.flatten(1).amin(1)).clamp_min(1e-12)
    c1 = ((0.01 * data_range) ** 2).view(-1, 1, 1, 1, 1)
    c2 = ((0.03 * data_range) ** 2).view(-1, 1, 1, 1, 1)
    mu_a, mu_b = mean(a), mean(b)
    var_a = mean(a * a) - mu_a ** 2
    var_b = mean(b * b) - mu_b ** 2
    cov = mean(a * b) - mu_a * mu_b
    ssim = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / \
        ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))
    return ssim.flatten(1).mean(1)


def evaluate(model, loader, pkg, device):
    """Relative l2 on the complex field and SSIM on its amplitude, over a loader."""
    import torch
    model.eval()
    rel, ssim, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in loader:
            x, y, aux = pkg.inputs(batch)
            x, y = x.to(device), y.to(device)
            pred = model(x).float()
            diff = (pred - y).flatten(1).norm(dim=1)
            rel += (diff / y.flatten(1).norm(dim=1).clamp_min(1e-12)).sum().item()
            amp_p = pred.pow(2).sum(1).sqrt()
            amp_y = y.pow(2).sum(1).sqrt()
            ssim += ssim3d(amp_p, amp_y).sum().item()
            n += y.shape[0]
    model.train()
    if n == 0:
        return {'val_samples': 0}
    return {'val_rel_l2': rel / n, 'val_ssim': ssim / n, 'val_samples': n}


def train_round(model, loader, pkg, device, cfg, stage, logger):
    """Run ``local_epochs`` over ``loader``; return training metrics for the round."""
    import torch
    epochs = int(cfg.get('local_epochs', 1))
    accum = max(1, int(cfg.get('grad_accum', 1)))
    clip = cfg.get('clip_norm')
    use_amp = bool(cfg.get('amp', True)) and device.type == 'cuda'
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(cfg.get('lr', 1e-3)),
                                  weight_decay=float(cfg.get('weight_decay', 0.0)))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    epoch_seconds, last_loss = [], None
    for epoch in range(epochs):
        start, total, count = time.time(), 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader, start=1):
            x, y, aux = pkg.inputs(batch)
            x, y, aux = x.to(device), y.to(device), _to(aux, device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                pred = model(x)
            loss, parts = pkg.loss(pred.float(), y, aux, stage)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    'non-finite loss at epoch {} step {}'.format(epoch + 1, step))
            if scaler:
                scaler.scale(loss / accum).backward()
            else:
                (loss / accum).backward()
            if step % accum == 0 or step == len(loader):
                if scaler:
                    scaler.unscale_(optimizer)
                if clip:
                    torch.nn.utils.clip_grad_norm_(params, float(clip))
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach())
            count += 1
        epoch_seconds.append(time.time() - start)
        last_loss = total / max(count, 1)
        logger.info('Epoch {} of {}: loss {:.5f}, {:.1f} s'.format(
            epoch + 1, epochs, last_loss, epoch_seconds[-1]))
    return {'train_loss': last_loss, 'epoch_seconds': [round(s, 2) for s in epoch_seconds],
            'peak_memory_mb': round(peak_memory_mb(device), 1), 'device': device.type,
            'amp': use_amp}
