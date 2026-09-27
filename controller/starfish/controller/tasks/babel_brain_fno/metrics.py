"""The paper's six evaluation metrics for BabelBrainFno, SF-05.

Field metrics: relative l2 error in percent on the complex field, and SSIM
and PSNR on the amplitude. Focal metrics, inside the brain mask only:
distance between predicted and true peak in mm, absolute relative error of
the peak amplitude in percent, and distance between the centroids of the
-3 dB focal regions in mm. Definitions follow section 2.2 of Tayeb's
preprint, context/tfus-fno-paper-notes.md in babelbrain-docs.

The paper does not pin every detail. Each open detail is a CONFIRM constant
below, to check against Tayeb's evaluation script, T7. The target is agreement
within 1e-4 relative on his saved test predictions.

Fields are arrays of shape (2, X, Y, Z), real and imaginary parts. Only
ratios and positions are used, so any common scaling of both fields, such as
the model package's normalisation, leaves every metric unchanged.
"""

import math

import numpy as np

# CONFIRM with Tayeb, T7: SSIM window. Here a uniform cube of this many voxels per side.
SSIM_WINDOW = 7
# CONFIRM, T7: SSIM and PSNR data range. Here the true amplitude's maximum minus its minimum.
DATA_RANGE = 'true_max_minus_min'
# CONFIRM, T7: peak amplitude error compares each field's own intra-brain peak.
PEAK_ERROR_AT = 'own_peak'
# CONFIRM, T7: the -3 dB focal region is every brain voxel at or above this share of
# the intra-brain peak amplitude. -3 dB in pressure amplitude is 10 ** (-3 / 20).
FOCAL_AMPLITUDE_FRACTION = 10 ** (-3 / 20)
# CONFIRM, T7: centroid of the region's voxel positions, not weighted by amplitude.
CENTROID_WEIGHTED = False

METRICS = ('rel_l2_pct', 'ssim', 'psnr_db', 'peak_distance_mm', 'peak_amplitude_error_pct',
           'focal_centroid_distance_mm')


def amplitude(field):
    field = np.asarray(field, dtype=np.float64)
    return np.sqrt(field[0] ** 2 + field[1] ** 2)


def relative_l2_pct(pred, true):
    pred, true = np.asarray(pred, np.float64), np.asarray(true, np.float64)
    return 100.0 * np.linalg.norm(pred - true) / max(np.linalg.norm(true), 1e-30)


def _data_range(true_amp):
    return max(float(true_amp.max() - true_amp.min()), 1e-30)


def ssim(pred_amp, true_amp, window=SSIM_WINDOW):
    """Mean SSIM of two 3D volumes with a uniform window, as in Wang et al. 2004."""
    from scipy.ndimage import uniform_filter
    a, b = np.asarray(pred_amp, np.float64), np.asarray(true_amp, np.float64)
    data_range = _data_range(b)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2

    def mean(t):
        return uniform_filter(t, size=window, mode='reflect')

    mu_a, mu_b = mean(a), mean(b)
    var_a = mean(a * a) - mu_a ** 2
    var_b = mean(b * b) - mu_b ** 2
    cov = mean(a * b) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / \
        ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))
    return float(s.mean())


def psnr_db(pred_amp, true_amp):
    mse = float(np.mean((np.asarray(pred_amp, np.float64) -
                np.asarray(true_amp, np.float64)) ** 2))
    if mse == 0:
        return math.inf
    return 20 * math.log10(_data_range(np.asarray(true_amp)) / math.sqrt(mse))


def _brain_peak(amp, mask):
    masked = np.where(mask, amp, -np.inf)
    index = np.unravel_index(int(np.argmax(masked)), amp.shape)
    return np.array(index, dtype=np.float64), float(amp[index])


def _focal_centroid(amp, mask, peak_value):
    region = mask & (amp >= FOCAL_AMPLITUDE_FRACTION * peak_value)
    points = np.argwhere(region).astype(np.float64)
    if CENTROID_WEIGHTED:
        weights = amp[region]
        return (points * weights[:, None]).sum(0) / weights.sum()
    return points.mean(0)


def sample_metrics(pred, true, brain_mask, spacing_mm):
    """The six metrics for one sample."""
    mask = np.asarray(brain_mask) > 0
    if not mask.any():
        raise ValueError('brain mask is empty')
    pred_amp, true_amp = amplitude(pred), amplitude(true)
    p_idx, p_peak = _brain_peak(pred_amp, mask)
    t_idx, t_peak = _brain_peak(true_amp, mask)
    spacing = float(spacing_mm)
    return {
        'rel_l2_pct': relative_l2_pct(pred, true),
        'ssim': ssim(pred_amp, true_amp),
        'psnr_db': psnr_db(pred_amp, true_amp),
        'peak_distance_mm': float(np.linalg.norm(p_idx - t_idx) * spacing),
        'peak_amplitude_error_pct': 100.0 * abs(p_peak - t_peak) / max(t_peak, 1e-30),
        'focal_centroid_distance_mm': float(np.linalg.norm(
            _focal_centroid(pred_amp, mask, p_peak) - _focal_centroid(true_amp, mask, t_peak))
            * spacing),
    }


def summarize(rows):
    """Mean, standard deviation and median of each metric over samples."""
    out = {'n': len(rows)}
    for name in METRICS:
        values = np.array([r[name] for r in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            out[name] = {'mean': None, 'std': None, 'median': None}
            continue
        out[name] = {'mean': float(finite.mean()), 'std': float(finite.std()),
                     'median': float(np.median(finite))}
    return out
