"""Utilities for uncertainty calibration, analysis, and refinement.

This module is intentionally independent from inference entrypoints so it can be
used from scripts, notebooks, or custom evaluation pipelines.
"""

from statistics import NormalDist
from typing import Dict, Iterable

import torch


def uncertainty_to_confidence(uncertainty: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Convert uncertainty to confidence in [0, 1], where brighter means more confidence."""
    uncertainty = uncertainty.float()
    u_min = uncertainty.amin(dim=(-2, -1), keepdim=True)
    u_max = uncertainty.amax(dim=(-2, -1), keepdim=True)
    normalized = (uncertainty - u_min) / (u_max - u_min).clamp_min(eps)
    return (1.0 - normalized).clamp(0.0, 1.0)


def normalize_uncertainty_map(
    uncertainty_map: torch.Tensor,
    mean_prediction: torch.Tensor,
    uncertainty_metric: str = "std",
    mode: str = "absolute",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalize uncertainty to reduce brightness-dependent scale effects."""
    if uncertainty_map.shape != mean_prediction.shape:
        raise ValueError("uncertainty_map and mean_prediction must have the same shape.")

    uncertainty_metric = uncertainty_metric.lower()
    mode = mode.lower()

    if uncertainty_metric not in {"std", "var"}:
        raise ValueError("uncertainty_metric must be 'std' or 'var'.")
    if mode not in {"absolute", "relative_mean"}:
        raise ValueError("mode must be 'absolute' or 'relative_mean'.")

    if mode == "absolute":
        return uncertainty_map

    scale = mean_prediction.abs().clamp_min(eps)
    if uncertainty_metric == "std":
        return uncertainty_map / scale
    return uncertainty_map / scale.pow(2)


def ensemble_statistics(
    ensemble_predictions: torch.Tensor,
    uncertainty_metric: str = "std",
) -> Dict[str, torch.Tensor]:
    """
    Compute ensemble moments.

    Args:
        ensemble_predictions: Tensor of shape (N, B, C, H, W).
        uncertainty_metric: "std" or "var".

    Returns:
        Dictionary with mean, std, var, and uncertainty map.
    """
    if ensemble_predictions.ndim != 5:
        raise ValueError("ensemble_predictions must have shape (N, B, C, H, W).")

    uncertainty_metric = uncertainty_metric.lower()
    if uncertainty_metric not in {"std", "var"}:
        raise ValueError("uncertainty_metric must be 'std' or 'var'.")

    mean_pred = ensemble_predictions.mean(dim=0)
    std_pred = ensemble_predictions.std(dim=0, unbiased=False)
    var_pred = ensemble_predictions.var(dim=0, unbiased=False)

    uncertainty = std_pred if uncertainty_metric == "std" else var_pred
    return {
        "mean": mean_pred,
        "std": std_pred,
        "var": var_pred,
        "uncertainty": uncertainty,
    }


def prediction_error(
    mean_prediction: torch.Tensor,
    hr: torch.Tensor,
    error_metric: str = "abs",
) -> torch.Tensor:
    """
    Pixelwise prediction error map from ensemble mean against target HR.

    Args:
        mean_prediction: (B, C, H, W)
        hr: (B, C, H, W)
        error_metric: "abs" or "squared".
    """
    if mean_prediction.shape != hr.shape:
        raise ValueError("mean_prediction and hr must have the same shape.")

    error_metric = error_metric.lower()
    if error_metric == "abs":
        return (mean_prediction - hr).abs()
    if error_metric == "squared":
        return (mean_prediction - hr).pow(2)
    raise ValueError("error_metric must be 'abs' or 'squared'.")


def _pearson_corr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    x = x.flatten().float()
    y = y.flatten().float()
    x = x - x.mean()
    y = y - y.mean()
    denom = (x.norm() * y.norm()).clamp_min(eps)
    return float(torch.dot(x, y) / denom)


def _rankdata_ordinal(x: torch.Tensor) -> torch.Tensor:
    # Ordinal ranking is robust and efficient; ties receive arbitrary but stable order.
    order = torch.argsort(x)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, x.numel() + 1, device=x.device, dtype=torch.float32)
    return ranks


def spearman_corr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    """Spearman correlation by Pearson on ordinal ranks."""
    xf = x.flatten().float()
    yf = y.flatten().float()
    xr = _rankdata_ordinal(xf)
    yr = _rankdata_ordinal(yf)
    return _pearson_corr(xr, yr, eps=eps)


def uncertainty_error_correlation(
    uncertainty_map: torch.Tensor,
    error_map: torch.Tensor,
) -> Dict[str, object]:
    """
    Correlate uncertainty and error globally and per channel.

    Args:
        uncertainty_map: (B, C, H, W)
        error_map: (B, C, H, W)
    """
    if uncertainty_map.shape != error_map.shape:
        raise ValueError("uncertainty_map and error_map must have the same shape.")

    _, channels, _, _ = uncertainty_map.shape
    per_channel = []
    for ch in range(channels):
        u_ch = uncertainty_map[:, ch]
        e_ch = error_map[:, ch]
        per_channel.append(
            {
                "channel": ch,
                "pearson": _pearson_corr(u_ch, e_ch),
                "spearman": spearman_corr(u_ch, e_ch),
            }
        )

    return {
        "pearson": _pearson_corr(uncertainty_map, error_map),
        "spearman": spearman_corr(uncertainty_map, error_map),
        "per_channel": per_channel,
    }


def calibration_coverage(
    mean_prediction: torch.Tensor,
    std_prediction: torch.Tensor,
    hr: torch.Tensor,
    interval_levels: Iterable[float] = (0.5, 0.8, 0.9, 0.95),
    eps: float = 1e-8,
) -> Dict[str, object]:
    """
    Evaluate interval calibration for Gaussian predictive distribution.

    For each coverage level p, use central interval:
        mean +/- z_p * std,
    where z_p = Phi^{-1}((1 + p) / 2).
    """
    if mean_prediction.shape != hr.shape or std_prediction.shape != hr.shape:
        raise ValueError("mean_prediction, std_prediction, and hr must have the same shape.")

    std_prediction = std_prediction.clamp_min(eps)
    _, channels, _, _ = mean_prediction.shape

    rows = []
    for p in interval_levels:
        if not (0.0 < p < 1.0):
            raise ValueError("interval levels must be in (0, 1).")

        z = NormalDist().inv_cdf((1.0 + float(p)) / 2.0)
        lower = mean_prediction - z * std_prediction
        upper = mean_prediction + z * std_prediction

        inside = (hr >= lower) & (hr <= upper)
        empirical = float(inside.float().mean())

        per_channel = []
        for ch in range(channels):
            per_channel.append(float(inside[:, ch].float().mean()))

        rows.append(
            {
                "interval": float(p),
                "z_value": float(z),
                "empirical": empirical,
                "gap": empirical - float(p),
                "per_channel": per_channel,
            }
        )

    return {"coverage": rows}


def gaussian_nll(
    mean_prediction: torch.Tensor,
    var_prediction: torch.Tensor,
    hr: torch.Tensor,
    eps: float = 1e-8,
) -> Dict[str, object]:
    """Compute pixelwise Gaussian NLL and aggregates."""
    if mean_prediction.shape != hr.shape or var_prediction.shape != hr.shape:
        raise ValueError("mean_prediction, var_prediction, and hr must have the same shape.")

    var_prediction = var_prediction.clamp_min(eps)
    sq_error = (hr - mean_prediction).pow(2)
    nll = 0.5 * (torch.log(2.0 * torch.pi * var_prediction) + sq_error / var_prediction)

    _, channels, _, _ = nll.shape
    per_channel = [float(nll[:, ch].mean()) for ch in range(channels)]

    return {
        "mean_nll": float(nll.mean()),
        "per_channel": per_channel,
        "nll_map": nll,
    }


def sparsification_curve(
    uncertainty_map: torch.Tensor,
    error_map: torch.Tensor,
    remove_fractions: Iterable[float] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
    eps: float = 1e-8,
) -> Dict[str, object]:
    """
    Evaluate how error changes when highest-uncertainty pixels are ignored.

    This is an analysis-only metric: it checks ranking quality of uncertainty.
    """
    if uncertainty_map.shape != error_map.shape:
        raise ValueError("uncertainty_map and error_map must have the same shape.")

    u = uncertainty_map.flatten().float()
    e = error_map.flatten().float()

    rows = []
    for frac in remove_fractions:
        frac = float(frac)
        if not (0.0 <= frac < 1.0):
            raise ValueError("remove_fractions entries must be in [0, 1).")

        keep_ratio = 1.0 - frac
        threshold = torch.quantile(u, keep_ratio)
        keep_mask = u <= threshold

        kept_count = int(keep_mask.sum().item())
        if kept_count == 0:
            kept_mae = float("nan")
            kept_rmse = float("nan")
        else:
            kept_err = e[keep_mask]
            kept_mae = float(kept_err.mean())
            kept_rmse = float(torch.sqrt(kept_err.pow(2).mean().clamp_min(eps)))

        rows.append(
            {
                "remove_fraction": frac,
                "keep_fraction": keep_ratio,
                "num_kept_pixels": kept_count,
                "mae": kept_mae,
                "rmse": kept_rmse,
            }
        )

    return {"curve": rows}


def confidence_guided_blend(
    mean_prediction: torch.Tensor,
    baseline_prediction: torch.Tensor,
    confidence_map: torch.Tensor,
    strength: float = 1.0,
) -> torch.Tensor:
    """
    Blend prediction with baseline based on confidence.

    blend = confidence^strength * mean_prediction + (1 - confidence^strength) * baseline
    """
    if mean_prediction.shape != baseline_prediction.shape:
        raise ValueError("mean_prediction and baseline_prediction must have the same shape.")

    if confidence_map.shape != mean_prediction.shape:
        raise ValueError("confidence_map must have the same shape as predictions.")

    if strength <= 0:
        raise ValueError("strength must be > 0.")

    c = confidence_map.clamp(0.0, 1.0).pow(strength)
    return c * mean_prediction + (1.0 - c) * baseline_prediction


def robust_ensemble_average(
    ensemble_predictions: torch.Tensor,
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Robust aggregation by downweighting per-pixel outlier members.

    Args:
        ensemble_predictions: (N, B, C, H, W)
    """
    if ensemble_predictions.ndim != 5:
        raise ValueError("ensemble_predictions must have shape (N, B, C, H, W).")
    if temperature <= 0:
        raise ValueError("temperature must be > 0.")

    median = ensemble_predictions.median(dim=0).values
    deviation = (ensemble_predictions - median.unsqueeze(0)).abs()

    mad = deviation.median(dim=0).values.clamp_min(eps)
    scaled = deviation / (temperature * mad.unsqueeze(0))

    weights = torch.exp(-scaled)
    weights = weights / weights.sum(dim=0, keepdim=True).clamp_min(eps)

    return (weights * ensemble_predictions).sum(dim=0)


def evaluate_uncertainty_calibration(
    mean_prediction: torch.Tensor,
    hr: torch.Tensor,
    uncertainty_map: torch.Tensor,
    std_prediction: torch.Tensor = None,
    var_prediction: torch.Tensor = None,
    interval_levels: Iterable[float] = (0.5, 0.8, 0.9, 0.95),
    error_metric: str = "abs",
    remove_fractions: Iterable[float] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
) -> Dict[str, object]:
    """
    One-call uncertainty evaluation bundle.

    Includes:
    - uncertainty/error correlation
    - sparsification curve
    - optional interval coverage (if std_prediction is provided)
    - optional Gaussian NLL (if var_prediction is provided)
    """
    error_map = prediction_error(mean_prediction, hr, error_metric=error_metric)

    result = {
        "correlation": uncertainty_error_correlation(uncertainty_map, error_map),
        "sparsification": sparsification_curve(
            uncertainty_map, error_map, remove_fractions=remove_fractions
        ),
    }

    if std_prediction is not None:
        result["coverage"] = calibration_coverage(
            mean_prediction,
            std_prediction,
            hr,
            interval_levels=interval_levels,
        )

    if var_prediction is not None:
        result["gaussian_nll"] = gaussian_nll(mean_prediction, var_prediction, hr)

    return result


def refine_prediction_with_uncertainty(
    mean_prediction: torch.Tensor,
    baseline_prediction: torch.Tensor,
    uncertainty_map: torch.Tensor,
    blend_strength: float = 1.0,
    use_confidence_normalization: bool = True,
    ensemble_runs: int = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Uncertainty-aware refinement by confidence-guided blending.

    If use_confidence_normalization=True, uncertainty is converted to confidence
    per image and channel. Otherwise uncertainty_map is interpreted as confidence.
    """
    confidence_map = (
        uncertainty_to_confidence(uncertainty_map, eps=eps)
        if use_confidence_normalization
        else uncertainty_map.clamp(0.0, 1.0)
    )

    effective_strength = float(blend_strength)
    if ensemble_runs is not None:
        if int(ensemble_runs) <= 0:
            raise ValueError("ensemble_runs must be > 0 when provided.")
        effective_strength = effective_strength / float(int(ensemble_runs))

    return confidence_guided_blend(
        mean_prediction,
        baseline_prediction,
        confidence_map,
        strength=effective_strength,
    )
